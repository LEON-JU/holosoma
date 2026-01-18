"""
Reconstruction + retarget visualization (customized).

This script visualizes:
- reconstructed scene point cloud (RGB-D fusion or fallback PLY),
- reconstructed SMPL-X human mesh,
- retargeted robot motion (qpos .npz),
in a single Viser viewer, reusing the pipeline artifacts saved under `holosoma_runs/`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import cv2
import open3d as o3d
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

from holosoma_retargeting import ground_alignment
from holosoma_retargeting.config import ManualPaths, RobotType, SeqName, get_sequence_paths
from holosoma_retargeting.src.viser_utils_recon import create_motion_control_sliders_with_callbacks


def _load_npz_qpos(npz_path: Path) -> tuple[np.ndarray, int]:
    data = np.load(str(npz_path), allow_pickle=True)
    qpos = data["qpos"]
    fps = int(data["fps"]) if "fps" in data.files else 30
    return qpos, fps


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _apply_display_transform(points: np.ndarray, *, z_min: float, scale: float, apply_z_min: bool, apply_scale: bool) -> np.ndarray:
    pts = points.copy()
    if apply_z_min:
        pts[..., 2] -= float(z_min)
    if apply_scale:
        pts *= float(scale)
    return pts


def _downsample(points: np.ndarray, colors: np.ndarray | None, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    if points.shape[0] <= max_points:
        return points, colors
    idx = np.random.choice(points.shape[0], size=max_points, replace=False)
    pts = points[idx]
    cols = colors[idx] if colors is not None else None
    return pts, cols


def _opencv_to_zup_rotation(R_cv: np.ndarray) -> np.ndarray:
    # Map OpenCV axes to Z-up: x->x, y->-z, z->y (row-wise mapping).
    M = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)
    return M @ R_cv


def _create_camera_frustum_lines(cam_pose: np.ndarray, size: float = 0.15):
    center = cam_pose[:3, 3]
    rotation = cam_pose[:3, :3]
    aspect = 4.0 / 3.0
    half_width = size * aspect
    half_height = size
    depth = size * 2
    corners_cam = np.array(
        [
            [half_width, half_height, depth],
            [-half_width, half_height, depth],
            [-half_width, -half_height, depth],
            [half_width, -half_height, depth],
        ]
    )
    corners_world = center + (rotation @ corners_cam.T).T
    lines = []
    for corner in corners_world:
        lines.append((center, corner))
    for i in range(4):
        lines.append((corners_world[i], corners_world[(i + 1) % 4]))
    return lines


@dataclass
class ReconViewerConfig:
    seq: SeqName
    robot: RobotType = "g1"
    manual: ManualPaths = ManualPaths()

    scene_source: Literal["rgbd", "ply"] = "rgbd"
    world_mode: Literal["human", "retarget", "retarget_no_zmin", "retarget_no_scale"] = "retarget"
    """human: aligned reconstruction; retarget: apply z_min + scale; retarget_no_zmin/retarget_no_scale toggle each."""

    qpos_npz: Optional[Path] = None
    robot_urdf: Optional[str] = None
    # Downsampling (remote-friendly defaults)
    scene_max_points: int = 150_000
    human_max_points: int = 20_000
    frame_max_points: int = 20_000


def main(cfg: ReconViewerConfig) -> None:
    paths = get_sequence_paths(seq=cfg.seq, robot=cfg.robot, manual=cfg.manual)
    run_summary = _load_json(paths.pipeline_config_json) if paths.pipeline_config_json.exists() else {}

    scale_factor = float(run_summary.get("scale_factor", 1.0))
    z_min_human = float(run_summary.get("vertical_bias_z_min_human_m", 0.0))
    T_align = np.asarray(_load_json(paths.ground_transform_json)["transform_matrix"], dtype=np.float64)

    qpos_path = cfg.qpos_npz or (paths.retarget_save_dir / f"{cfg.seq}.npz")
    qpos, fps = _load_npz_qpos(qpos_path)

    # Load scene (OpenCV-ish), convert to Z-up, then apply ground alignment transform.
    if cfg.scene_source == "rgbd":
        scene_points_cv, scene_colors = ground_alignment.load_scene_pointcloud_from_rgbd(
            data_dir=str(paths.predicted_dir),
            depth_dir=str(paths.depth_recovered),
            smplx_results_dir=str(paths.results_dir),
        )
        scene_points = ground_alignment.opencv_to_z_up(scene_points_cv)
        scene_points = ground_alignment.apply_transform_matrix(scene_points, T_align)
    else:
        pcd = o3d.io.read_point_cloud(str(paths.aligned_scene_ply))
        scene_points = np.asarray(pcd.points, dtype=np.float64)
        scene_colors = np.asarray(pcd.colors, dtype=np.float32)
    if scene_colors is None or scene_colors.size == 0:
        scene_colors = np.full((scene_points.shape[0], 3), 0.7, dtype=np.float32)
    scene_points_base = scene_points.copy()
    scene_colors_base = scene_colors.copy() if scene_colors is not None else None

    # Human mesh: base world (Z-up) then apply alignment.
    human_vertices = ground_alignment.load_human_mesh_vertices_base(str(paths.all_results_video), str(paths.depth_recovered))
    human_faces = None
    if human_vertices is not None:
        human_faces = ground_alignment.load_smplx_faces(str(cfg.manual.smpl_model_path))
        human_vertices = np.stack([ground_alignment.apply_transform_matrix(v, T_align) for v in human_vertices], axis=0)
    human_vertices_base = human_vertices.copy() if human_vertices is not None else None

    scene_points = _apply_display_transform(
        scene_points_base, z_min=0, scale=scale_factor, apply_z_min=False, apply_scale=True
    )
    if human_vertices_base is not None:
        human_vertices = _apply_display_transform(
            human_vertices_base, z_min=z_min_human, scale=scale_factor, apply_z_min=True, apply_scale=True
        )

    # Viewer
    server = viser.ViserServer()
    server.scene.add_frame("/world_axes", show_axes=True)

    # Scene point cloud
    scene_points, scene_colors = _downsample(scene_points, scene_colors, cfg.scene_max_points)
    colors_uint8 = (np.clip(scene_colors, 0.0, 1.0) * 255.0).astype(np.uint8) if scene_colors is not None else None
    scene_handle = server.scene.add_point_cloud(
        "/scene/cloud",
        points=scene_points.astype(np.float32),
        colors=colors_uint8,
        point_size=0.02,
        visible=True,
    )

    # Human mesh
    human_handle = {"mesh": None}
    if human_vertices is not None and human_faces is not None:
        server.scene.add_frame("/human", show_axes=False)
        human_handle["mesh"] = server.scene.add_mesh_simple(
            "/human/mesh",
            vertices=human_vertices[0].astype(np.float32),
            faces=human_faces,
            color=(200, 200, 200),
            opacity=0.7,
        )
    # Human point cloud (downsampled per frame)
    human_pc_handle = {"pc": None}

    # Robot
    if cfg.robot_urdf is not None:
        robot_urdf_path = cfg.robot_urdf
    else:
        robot_urdf_path = "holosoma_retargeting/models/g1/g1_29dof.urdf" if cfg.robot == "g1" else "holosoma_retargeting/models/t1/t1_23dof.urdf"
    robot_urdf = yourdfpy.URDF.load(robot_urdf_path, load_meshes=True, build_scene_graph=True)
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    vr = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robot")
    robot_dof = len(vr.get_actuated_joint_limits())

    # Per-frame projected cloud state
    frame_cloud_handle = {"pc": None}
    camera_frame_handle = {"frame": None}

    avg_scene_scale = ground_alignment.load_scale_factor(str(paths.depth_recovered), str(paths.all_results_video))

    @lru_cache(maxsize=256)
    def _load_frame_cloud(frame_idx: int, keep_human: bool) -> tuple[np.ndarray, np.ndarray] | None:
        frame_id = f"{frame_idx:04d}"
        rgb_file = paths.predicted_dir / f"{frame_id}_color.png"
        depth_file = paths.predicted_dir / f"{frame_id}_depth.png"
        if not rgb_file.exists() or not depth_file.exists():
            return None

        rgb_image = cv2.imread(str(rgb_file))[:, :, ::-1]
        depth_map_m = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

        mask_file = paths.predicted_dir / f"{frame_id}_mask.png"
        if mask_file.exists():
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if keep_human:
                keep_mask = (mask == 0).astype(np.uint8)
            else:
                keep_mask = (mask > 0).astype(np.uint8)
        else:
            keep_mask = None

        pose_file = paths.predicted_dir / f"{frame_id}_pose.txt"
        if not pose_file.exists():
            pose_file = paths.predicted_dir / f"{frame_id}_pose.pi3.txt"
        if not pose_file.exists():
            return None

        try:
            camera_pose = np.loadtxt(str(pose_file), delimiter=",")
        except ValueError:
            camera_pose = np.loadtxt(str(pose_file), delimiter=" ")

        smplx_file = paths.results_dir / frame_id / "smplx_params.pt"
        if not smplx_file.exists():
            return None

        smplx_data = ground_alignment.torch.load(str(smplx_file), map_location="cpu", weights_only=False)  # type: ignore[attr-defined]
        cam_int = smplx_data["cam_int"].numpy()

        depth_map_scaled = depth_map_m * float(avg_scene_scale)
        camera_pose_scaled = camera_pose.copy()
        camera_pose_scaled[:3, 3] = camera_pose[:3, 3] * float(avg_scene_scale)

        pts, cols = ground_alignment.create_pointcloud_from_rgbd(
            rgb_image, depth_map_scaled, cam_int, camera_pose_scaled, keep_mask
        )

        pts = ground_alignment.opencv_to_z_up(pts)
        pts = ground_alignment.apply_transform_matrix(pts, T_align)
        cols = cols.astype(np.float32) / 255.0
        return pts, cols

    def _update_camera_frame(frame_idx: int) -> None:
        frame_id = f"{frame_idx:04d}"
        pose_file = paths.predicted_dir / f"{frame_id}_pose.txt"
        if not pose_file.exists():
            pose_file = paths.predicted_dir / f"{frame_id}_pose.pi3.txt"
        if not pose_file.exists():
            return
        try:
            camera_pose = np.loadtxt(str(pose_file), delimiter=",")
        except ValueError:
            camera_pose = np.loadtxt(str(pose_file), delimiter=" ")
        camera_pose[:3, 3] = camera_pose[:3, 3] * float(avg_scene_scale)
        cam_pos = ground_alignment.opencv_to_z_up(camera_pose[:3, 3].reshape(1, 3)).reshape(3)
        cam_pos = ground_alignment.apply_transform_matrix(cam_pos[None, :], T_align)[0]
        cam_pos = _apply_display_transform(
            cam_pos[None, :], z_min=0.0, scale=scale_factor, apply_z_min=False, apply_scale=True
        )[0]
        if camera_frame_handle["frame"] is None:
            camera_frame_handle["frame"] = server.scene.add_frame(
                "/camera/frame",
                show_axes=True,
                position=cam_pos,
                axes_length=0.1,
            )
        else:
            camera_frame_handle["frame"].position = cam_pos

    def on_frame(i: int) -> None:
        if human_handle["mesh"] is not None and human_vertices is not None:
            human_handle["mesh"].vertices = human_vertices[i].astype(np.float32)

        if human_pc_handle["pc"] is not None:
            human_pts = _load_frame_cloud(i, keep_human=True)
            if human_pts is not None:
                pts_raw, cols_raw = human_pts
                pts = _apply_display_transform(
                    pts_raw, z_min=0.0, scale=scale_factor, apply_z_min=False, apply_scale=True
                )
                pts, cols = _downsample(pts, cols_raw, cfg.human_max_points)
                human_pc_handle["pc"].points = pts.astype(np.float32)
                if cols is not None:
                    human_pc_handle["pc"].colors = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)

        if frame_cloud_handle["pc"] is not None:
            frame_pts = _load_frame_cloud(i, keep_human=False)
            if frame_pts is not None:
                pts_raw, cols_raw = frame_pts
                pts = _apply_display_transform(
                    pts_raw, z_min=0, scale=scale_factor, apply_z_min=False, apply_scale=True
                )
                pts, cols = _downsample(pts, cols_raw, cfg.frame_max_points)
                frame_cloud_handle["pc"].points = pts.astype(np.float32)
                if cols is not None:
                    frame_cloud_handle["pc"].colors = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8)

        if camera_frame_handle["frame"] is not None:
            _update_camera_frame(i)

    create_motion_control_sliders_with_callbacks(
        server=server,
        viser_robot=vr,
        robot_base_frame=robot_root,
        motion_sequence=qpos,
        robot_dof=robot_dof,
        initial_fps=fps,
        initial_interp_mult=2,
        loop=True,
        on_frame=on_frame,
    )

    # ---------- UI toggles ----------
    with server.gui.add_folder("Scene / Mesh"):
        show_scene_cb = server.gui.add_checkbox("Show scene cloud", initial_value=True)
        show_frame_cb = server.gui.add_checkbox("Show frame cloud", initial_value=False)
        show_human_mesh_cb = server.gui.add_checkbox("Show human mesh", initial_value=human_handle["mesh"] is not None)
        show_human_pc_cb = server.gui.add_checkbox("Show human point cloud", initial_value=False)
        show_robot_mesh_cb = server.gui.add_checkbox("Show robot meshes", initial_value=True)
        show_camera_cb = server.gui.add_checkbox("Show camera", initial_value=False)

    @show_scene_cb.on_update
    def _(_evt):
        scene_handle.visible = bool(show_scene_cb.value)

    @show_frame_cb.on_update
    def _(_evt):
        if show_frame_cb.value:
            if frame_cloud_handle["pc"] is None:
                frame_cloud_handle["pc"] = server.scene.add_point_cloud(
                    "/scene/frame_cloud",
                    points=np.zeros((1, 3), dtype=np.float32),
                    colors=np.zeros((1, 3), dtype=np.uint8),
                    point_size=0.02,
                    visible=True,
                )
            else:
                frame_cloud_handle["pc"].visible = True
            on_frame(0)
        else:
            if frame_cloud_handle["pc"] is not None:
                frame_cloud_handle["pc"].visible = False

    @show_human_mesh_cb.on_update
    def _(_evt):
        if human_handle["mesh"] is not None:
            human_handle["mesh"].visible = bool(show_human_mesh_cb.value)

    @show_human_pc_cb.on_update
    def _(_evt):
        if show_human_pc_cb.value:
            if human_pc_handle["pc"] is None:
                human_pts = _load_frame_cloud(0, keep_human=True)
                if human_pts is None:
                    return
                pts_raw, cols_raw = human_pts
                pts = _apply_display_transform(
                    pts_raw, z_min=0.0, scale=scale_factor, apply_z_min=False, apply_scale=True
                )
                pts, cols = _downsample(pts, cols_raw, cfg.human_max_points)
                colors = (np.clip(cols, 0.0, 1.0) * 255.0).astype(np.uint8) if cols is not None else None
                human_pc_handle["pc"] = server.scene.add_point_cloud(
                    "/human/points",
                    points=pts.astype(np.float32),
                    colors=colors,
                    point_size=0.01,
                    visible=True,
                )
            elif human_pc_handle["pc"] is not None:
                human_pc_handle["pc"].visible = True
            on_frame(0)
        else:
            if human_pc_handle["pc"] is not None:
                human_pc_handle["pc"].visible = False

    @show_robot_mesh_cb.on_update
    def _(_evt):
        vr.show_visual = bool(show_robot_mesh_cb.value)

    @show_camera_cb.on_update
    def _(_evt):
        if show_camera_cb.value:
            _update_camera_frame(0)
            if camera_frame_handle["frame"] is not None:
                camera_frame_handle["frame"].visible = True
        else:
            if camera_frame_handle["frame"] is not None:
                camera_frame_handle["frame"].visible = False

    print("[viser_player_recon] Open the viewer URL printed above. Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(ReconViewerConfig))
