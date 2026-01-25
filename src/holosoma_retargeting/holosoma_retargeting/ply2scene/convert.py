from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import tyro

from holosoma_retargeting import ground_alignment
from holosoma_retargeting.config import ManualPaths, RobotType, SeqName, get_sequence_paths


@dataclass
class Ply2SceneConfig:
    seq: SeqName
    robot: RobotType = "g1"
    manual: ManualPaths = ManualPaths()

    output_dir: Optional[Path] = None
    """Defaults to <run_dir>/artifacts/ply2scene/scene."""

    # Point cloud generation
    max_points: int = 2_000_000
    roi_half_extent_m: float = 1.0
    depth_gradient_threshold_m: float = 0.05

    # Morphological operations for mask edge filtering
    morphology_kernel_size: int = 5
    """Size of the rectangular kernel for morphological operations on mask edges."""

    # Density control + normals
    voxel_size: float = 0.1
    voxel_max_points_per_cell: int = 20
    normal_radius: float = 0.2
    normal_max_nn: int = 30
    orient_normals_k: int = 30

    # Reconstruction
    use_ball_pivoting: bool = False
    poisson_depth_coarse: int = 7
    poisson_depth_final: int = 9
    poisson_density_quantile: float = 0.02
    bpa_radii: tuple[float, float, float] = (0.03, 0.06, 0.12)

    # Post-processing
    crop_to_aabb: bool = True

    # Scale handling
    mesh_scale_factor: Optional[float] = None
    """If None, read scale_factor from pipeline_config_json; used in URDF/MJCF mesh scale tags."""

    # MJCF includes
    write_includes: bool = False


def _load_transform_json(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    T = np.asarray(data["transform_matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"transform_matrix in {path} must be 4x4, got {T.shape}")
    return T


def _load_scale_factor(path: Path) -> float:
    if not path.exists():
        return 1.0
    with open(path, "r") as f:
        data = json.load(f)
    return float(data.get("scale_factor", 1.0))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _depth_gradient_mask(depth_m: np.ndarray, threshold: float) -> np.ndarray:
    if depth_m.ndim != 2:
        return np.ones_like(depth_m, dtype=bool)
    dy, dx = np.gradient(depth_m)
    grad = np.sqrt(dx * dx + dy * dy)
    return grad < float(threshold)


def _voxel_cap_sample(
    points: np.ndarray, colors: np.ndarray | None, voxel_size: float, max_per_voxel: int
) -> tuple[np.ndarray, np.ndarray | None]:
    if points.size == 0:
        return points, colors
    voxel_coords = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
    keep_indices: list[int] = []
    for voxel_id in np.unique(inverse):
        idx = np.where(inverse == voxel_id)[0]
        if idx.size <= max_per_voxel:
            keep_indices.extend(idx.tolist())
        else:
            chosen = np.random.choice(idx, size=int(max_per_voxel), replace=False)
            keep_indices.extend(chosen.tolist())
    keep = np.asarray(keep_indices, dtype=np.int64)
    pts = points[keep]
    cols = colors[keep] if colors is not None else None
    return pts, cols


def _points_from_rgbd(paths, T_align: np.ndarray, cfg: Ply2SceneConfig) -> tuple[np.ndarray, np.ndarray | None]:
    avg_scene_scale = ground_alignment.load_scale_factor(str(paths.depth_recovered), str(paths.all_results_video))
    frame_dirs = sorted(paths.results_dir.glob("[0-9][0-9][0-9][0-9]"))

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for frame_dir in frame_dirs:
        frame_id = frame_dir.name
        rgb_file = paths.predicted_dir / f"{frame_id}_color.png"
        depth_file = paths.predicted_dir / f"{frame_id}_depth.png"
        mask_file = paths.predicted_dir / f"{frame_id}_mask.png"
        pose_file = paths.predicted_dir / f"{frame_id}_pose.txt"
        if not pose_file.exists():
            pose_file = paths.predicted_dir / f"{frame_id}_pose.pi3.txt"

        if not rgb_file.exists() or not depth_file.exists() or not pose_file.exists():
            continue

        smplx_file = frame_dir / "smplx_params.pt"
        if not smplx_file.exists():
            continue

        rgb_image = cv2.imread(str(rgb_file))[:, :, ::-1]
        depth_map_m = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        try:
            camera_pose = np.loadtxt(str(pose_file), delimiter=",")
        except ValueError:
            camera_pose = np.loadtxt(str(pose_file), delimiter=" ")

        smplx_data = ground_alignment.torch.load(str(smplx_file), map_location="cpu", weights_only=False)  # type: ignore[attr-defined]
        cam_int = smplx_data["cam_int"].numpy()
        vertices = np.asarray(smplx_data.get("vertices"), dtype=np.float64)

        depth_map_scaled = depth_map_m * float(avg_scene_scale)
        camera_pose_scaled = camera_pose.copy()
        camera_pose_scaled[:3, 3] = camera_pose[:3, 3] * float(avg_scene_scale)

        grad_mask = _depth_gradient_mask(depth_map_scaled, cfg.depth_gradient_threshold_m)
        valid_mask = (depth_map_scaled > 0) & grad_mask

        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            scene_keep_mask = mask > 0
            # Apply morphological erosion to remove noisy edge pixels
            if cfg.morphology_kernel_size > 1:
                kernel = np.ones((cfg.morphology_kernel_size, cfg.morphology_kernel_size), np.uint8)
                scene_keep_mask = cv2.erode(scene_keep_mask.astype(np.uint8), kernel) > 0
            if scene_keep_mask is not None:
                scene_keep_mask = scene_keep_mask & valid_mask
        else:
            scene_keep_mask = valid_mask

        pts, cols = ground_alignment.create_pointcloud_from_rgbd(
            rgb_image, depth_map_scaled, cam_int, camera_pose_scaled, scene_keep_mask
        )

        if pts.size == 0:
            continue

        pts = ground_alignment.opencv_to_z_up(pts)
        pts = ground_alignment.apply_transform_matrix(pts, T_align)

        if vertices.size > 0:
            v_world = ground_alignment.transform_smplx_to_world(vertices, camera_pose_scaled, scale_factor=1.0)
            v_world = ground_alignment.opencv_to_z_up(v_world)
            v_world = ground_alignment.apply_transform_matrix(v_world, T_align)
            center = np.median(v_world, axis=0)
        else:
            center = np.median(pts, axis=0)

        extent = float(cfg.roi_half_extent_m)
        keep = np.all(np.abs(pts - center[None, :]) <= extent, axis=1)
        pts = pts[keep]
        cols = cols[keep] if cols is not None else None

        if pts.size == 0:
            continue

        all_points.append(pts)
        if cols is not None:
            all_colors.append(cols.astype(np.float32) / 255.0)

    if not all_points:
        raise ValueError("No valid frames found for point cloud generation.")

    points = np.vstack(all_points)
    colors = np.vstack(all_colors) if all_colors else None

    points, colors = _voxel_cap_sample(points, colors, cfg.voxel_size, int(cfg.voxel_max_points_per_cell))

    if points.shape[0] > cfg.max_points:
        idx = np.random.choice(points.shape[0], size=cfg.max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    return points.astype(np.float64), colors


def _prepare_point_cloud(
    points: np.ndarray, colors: np.ndarray | None, cfg: Ply2SceneConfig
) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and colors.size > 0:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(cfg.normal_radius),
            max_nn=int(cfg.normal_max_nn),
        )
    )
    if cfg.orient_normals_k > 0:
        pcd.orient_normals_consistent_tangent_plane(int(cfg.orient_normals_k))
    return pcd


def _reconstruct_mesh(pcd: o3d.geometry.PointCloud, cfg: Ply2SceneConfig, *, depth: int) -> o3d.geometry.TriangleMesh:
    if cfg.use_ball_pivoting:
        radii = o3d.utility.DoubleVector([float(r) for r in cfg.bpa_radii])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        densities = None
    else:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))

    if densities is not None:
        densities = np.asarray(densities)
        threshold = float(np.quantile(densities, cfg.poisson_density_quantile))
        keep = densities >= threshold
        mesh = mesh.select_by_index(np.where(keep)[0])

    if cfg.crop_to_aabb:
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def _write_urdf(path: Path, scale: float) -> None:
    urdf = f"""<?xml version="1.0"?>
<robot name="scene">
  <link name="scene_link">
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="meshes/scene_visual.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="meshes/scene_collision.obj" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    path.write_text(urdf)


def _write_mjcf(path: Path, scale: float) -> None:
    xml = f"""<mujoco model="scene">
  <compiler meshdir="meshes"/>

  <asset>
    <mesh name="scene_visual" file="scene_visual.obj" scale="{scale} {scale} {scale}"/>
    <mesh name="scene_collision" file="scene_collision.obj" scale="{scale} {scale} {scale}"/>
  </asset>

  <worldbody>
    <body name="scene">
      <geom name="scene_collision" type="mesh" mesh="scene_collision" contype="1" conaffinity="1" rgba="0 0 0 0"/>
      <geom name="scene_visual" type="mesh" mesh="scene_visual" contype="0" conaffinity="0" rgba="0.7 0.7 0.7 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml)


def _write_includes(assets_path: Path, body_path: Path, scale: float) -> None:
    assets_xml = f"""<mujocoinclude>
  <mesh name="scene_visual" file="meshes/scene_visual.obj" scale="{scale} {scale} {scale}"/>
  <mesh name="scene_collision" file="meshes/scene_collision.obj" scale="{scale} {scale} {scale}"/>
</mujocoinclude>
"""
    body_xml = """<mujocoinclude>
  <body name="scene">
    <geom name="scene_collision" type="mesh" mesh="scene_collision" contype="1" conaffinity="1" rgba="0 0 0 0"/>
    <geom name="scene_visual" type="mesh" mesh="scene_visual" contype="0" conaffinity="0" rgba="0.7 0.7 0.7 1"/>
  </body>
</mujocoinclude>
"""
    assets_path.write_text(assets_xml)
    body_path.write_text(body_xml)


def main(cfg: Ply2SceneConfig) -> None:
    paths = get_sequence_paths(seq=cfg.seq, robot=cfg.robot, manual=cfg.manual)
    T_align = _load_transform_json(paths.ground_transform_json)

    output_dir = cfg.output_dir or (paths.run_dir / "artifacts" / "ply2scene" / "scene")
    meshes_dir = output_dir / "meshes"
    _ensure_dir(meshes_dir)

    scale = cfg.mesh_scale_factor
    if scale is None:
        scale = _load_scale_factor(paths.pipeline_config_json)

    points, colors = _points_from_rgbd(paths, T_align, cfg)
    pcd = _prepare_point_cloud(points, colors, cfg)

    mesh_visual = _reconstruct_mesh(pcd, cfg, depth=int(cfg.poisson_depth_coarse))
    mesh_collision = mesh_visual

    visual_path = meshes_dir / "scene_visual.obj"
    collision_path = meshes_dir / "scene_collision.obj"
    o3d.io.write_triangle_mesh(str(visual_path), mesh_visual, write_triangle_uvs=False)
    o3d.io.write_triangle_mesh(str(collision_path), mesh_collision, write_triangle_uvs=False)

    _write_urdf(output_dir / "scene.urdf", float(scale))
    _write_mjcf(output_dir / "scene.xml", float(scale))

    if cfg.write_includes:
        _write_includes(output_dir / "scene_assets.xml", output_dir / "scene_body.xml", float(scale))

    meta = {
        "seq": cfg.seq,
        "robot": cfg.robot,
        "output_dir": str(output_dir),
        "mesh_scale_factor": float(scale),
        "paths": {
            "predicted_dir": str(paths.predicted_dir),
            "depth_recovered": str(paths.depth_recovered),
            "results_dir": str(paths.results_dir),
            "ground_transform_json": str(paths.ground_transform_json),
            "pipeline_config_json": str(paths.pipeline_config_json),
        },
        "points_count": int(points.shape[0]),
        "mesh_visual_triangles": int(len(mesh_visual.triangles)),
        "mesh_collision_triangles": int(len(mesh_collision.triangles)),
        "config": asdict(cfg),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"[ply2scene] Wrote scene to: {output_dir}")
    print(f"[ply2scene] Visual mesh: {visual_path}")
    print(f"[ply2scene] Collision mesh: {collision_path}")
    print(f"[ply2scene] URDF: {output_dir / 'scene.urdf'}")
    print(f"[ply2scene] MJCF: {output_dir / 'scene.xml'}")


if __name__ == "__main__":
    main(tyro.cli(Ply2SceneConfig))
