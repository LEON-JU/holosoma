"""
Automatic ground alignment using GeoCalib gravity estimation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
import tyro
import viser

from holosoma_retargeting import ground_alignment
from holosoma_retargeting.config import ManualPaths, RobotType, SeqName, get_sequence_paths
from holosoma_retargeting.postprocess.geocalib_ground_alignment import (
    GeoCalibGroundAlignmentConfig,
    compute_geocalib_ground_alignment,
    save_transform_json,
)


@dataclass(frozen=True)
class AutoGroundAlignArgs:
    seq: SeqName
    robot: RobotType = "g1"
    manual: ManualPaths = ManualPaths()
    use_rgbd_scene: bool = True

    num_geocalib_frames: int = 12
    geocalib_frame_stride: int = 5
    geocalib_frame_indices: Optional[str] = None
    geocalib_weights: str = "pinhole"
    geocalib_camera_y_up: bool = False
    geocalib_device: Optional[str] = None
    geocalib_angle_outlier_deg: float = 15.0
    debug: bool = False

    max_points: int = 200_000
    voxel_size: float = 0.05
    plane_distance_threshold: float = 0.03
    plane_ransac_n: int = 3
    plane_num_iterations: int = 2000
    plane_angle_deg: float = 10.0
    plane_max_candidates: int = 10
    recenter_xy: bool = True

    save_transform_json: Optional[Path] = None
    save_aligned_ply: Optional[Path] = None
    vis: bool = True
    exit_on_save: bool = False
    show_human: bool = False
    height_offset_range: float = 0.2


def _create_reference_plane(server: viser.ViserServer) -> viser.MeshHandle:
    plane_size = 20.0
    vertices = np.array(
        [
            [-plane_size, -plane_size, 0],
            [plane_size, -plane_size, 0],
            [plane_size, plane_size, 0],
            [-plane_size, plane_size, 0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return server.scene.add_mesh_simple(
        "/reference_plane",
        vertices=vertices,
        faces=faces,
        color=(200, 200, 200),
        opacity=1.0,
        wireframe=False,
    )


def _add_arrow(
    server: viser.ViserServer,
    name: str,
    origin: np.ndarray,
    vec: np.ndarray,
    *,
    color: tuple[int, int, int],
    line_width: float = 4.0,
    tip_radius: float = 0.03,
) -> tuple[viser.SplineCatmullRomHandle, viser.IcosphereHandle]:
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    tip = origin + vec
    line = server.scene.add_spline_catmull_rom(
        f"{name}/shaft",
        positions=np.stack([origin, tip], axis=0).astype(np.float32),
        color=color,
        line_width=float(line_width),
        visible=True,
    )
    tip_handle = server.scene.add_icosphere(
        f"{name}/tip",
        radius=float(tip_radius),
        position=tuple(tip.astype(float)),
        color=color,
    )
    return line, tip_handle


def _save_aligned_ply(path: Path, points: np.ndarray, colors_uint8: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors_uint8 is not None:
        colors_float = colors_uint8.astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_float)
    o3d.io.write_point_cloud(str(path), pcd)


def main(args: AutoGroundAlignArgs) -> None:
    paths = get_sequence_paths(seq=args.seq, robot=args.robot, manual=args.manual)

    cfg = GeoCalibGroundAlignmentConfig(
        predicted_dir=paths.predicted_dir,
        depth_dir=paths.depth_recovered,
        results_dir=paths.results_dir,
        fused_scene_ply=paths.fused_scene_ply,
        use_rgbd_scene=args.use_rgbd_scene,
        num_geocalib_frames=args.num_geocalib_frames,
        geocalib_frame_stride=args.geocalib_frame_stride,
        geocalib_frame_indices=args.geocalib_frame_indices,
        geocalib_weights=args.geocalib_weights,
        geocalib_camera_y_up=args.geocalib_camera_y_up,
        geocalib_device=args.geocalib_device,
        geocalib_angle_outlier_deg=args.geocalib_angle_outlier_deg,
        max_points=args.max_points,
        voxel_size=args.voxel_size,
        plane_distance_threshold=args.plane_distance_threshold,
        plane_ransac_n=args.plane_ransac_n,
        plane_num_iterations=args.plane_num_iterations,
        plane_angle_deg=args.plane_angle_deg,
        plane_max_candidates=args.plane_max_candidates,
        recenter_xy=args.recenter_xy,
    )
    res = compute_geocalib_ground_alignment(cfg, return_points=True, debug=args.debug)
    T_align = res.T_align
    z_up_dir = res.z_up_dir
    points_zup = res.points_zup
    colors_uint8 = res.colors_uint8
    points_zup_ds = res.points_zup_ds
    inliers = res.inliers_ds
    if points_zup is None or points_zup_ds is None or inliers is None:
        raise RuntimeError("Internal error: expected return_points=True payloads.")

    if args.save_transform_json is None:
        args_save = paths.ground_alignment_dir / "transform_auto.json"
    else:
        args_save = args.save_transform_json

    if not args.vis:
        save_transform_json(args_save, T_align, extra={"seq": args.seq, "robot": args.robot})
        if args.save_aligned_ply is not None:
            aligned_all = ground_alignment.apply_transform_matrix(points_zup, T_align)
            _save_aligned_ply(args.save_aligned_ply, aligned_all, colors_uint8)
        return

    server = viser.ViserServer()
    server.scene.add_frame("/world_axes", show_axes=True, position=(0.0, 0.0, 0.0))

    colors_original = colors_uint8 if colors_uint8 is not None else None
    pc_original = server.scene.add_point_cloud(
        name="/cloud/original",
        points=points_zup.astype(np.float32),
        colors=colors_original,
        point_size=0.02,
        visible=True,
    )

    aligned_all = ground_alignment.apply_transform_matrix(points_zup, T_align)
    pc_aligned = server.scene.add_point_cloud(
        name="/cloud/aligned",
        points=aligned_all.astype(np.float32),
        colors=colors_original,
        point_size=0.02,
        visible=False,
    )

    points_aligned_ds = ground_alignment.apply_transform_matrix(points_zup_ds, T_align)
    inlier_handle = server.scene.add_point_cloud(
        name="/cloud/ground_inliers",
        points=points_aligned_ds[inliers].astype(np.float32),
        colors=np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (inliers.shape[0], 1)),
        point_size=0.04,
        visible=True,
    )

    reference_plane = _create_reference_plane(server)
    reference_plane.visible = True

    gravity_len = 1.0
    gravity_arrow = _add_arrow(
        server,
        "/gravity/estimated_up",
        origin=np.zeros(3),
        vec=z_up_dir * gravity_len,
        color=(0, 255, 255),
        line_width=4.0,
        tip_radius=0.03,
    )
    rotated_arrow = _add_arrow(
        server,
        "/gravity/rotated_estimate",
        origin=np.zeros(3),
        vec=(T_align[:3, :3] @ z_up_dir) * gravity_len,
        color=(255, 165, 0),
        line_width=3.0,
        tip_radius=0.025,
    )
    z_axis_arrow = _add_arrow(
        server,
        "/gravity/z_axis",
        origin=np.zeros(3),
        vec=np.array([0.0, 0.0, gravity_len]),
        color=(255, 255, 0),
        line_width=3.0,
        tip_radius=0.025,
    )

    height_offset = 0.0

    def _current_T() -> np.ndarray:
        T = T_align.copy()
        T[2, 3] += height_offset
        return T

    def _update_aligned() -> None:
        T = _current_T()
        aligned_all_local = ground_alignment.apply_transform_matrix(points_zup, T)
        pc_aligned.points = aligned_all_local.astype(np.float32)
        points_aligned_local = ground_alignment.apply_transform_matrix(points_zup_ds, T)
        inlier_handle.points = points_aligned_local[inliers].astype(np.float32)

    with server.gui.add_folder("Display"):
        show_original = server.gui.add_checkbox("Show original", True)
        show_aligned = server.gui.add_checkbox("Show aligned", False)
        show_plane = server.gui.add_checkbox("Show z=0 plane", True)
        show_inliers = server.gui.add_checkbox("Show ground inliers", True)
        show_gravity = server.gui.add_checkbox("Show gravity vectors", True)

    @show_original.on_update
    def _(_evt):
        pc_original.visible = show_original.value

    @show_aligned.on_update
    def _(_evt):
        pc_aligned.visible = show_aligned.value

    @show_plane.on_update
    def _(_evt):
        reference_plane.visible = show_plane.value

    @show_inliers.on_update
    def _(_evt):
        inlier_handle.visible = show_inliers.value

    @show_gravity.on_update
    def _(_evt):
        for h in gravity_arrow + rotated_arrow + z_axis_arrow:
            h.visible = show_gravity.value

    with server.gui.add_folder("Height"):
        height_slider = server.gui.add_slider(
            "Height offset",
            min=-abs(args.height_offset_range),
            max=abs(args.height_offset_range),
            step=0.002,
            initial_value=0.0,
        )

    @height_slider.on_update
    def _(_evt):
        nonlocal height_offset
        height_offset = float(height_slider.value)
        _update_aligned()

    save_btn = server.gui.add_button("Save transform.json")

    @save_btn.on_click
    def _(_evt):
        T = _current_T()
        save_transform_json(args_save, T, extra={"seq": args.seq, "robot": args.robot})
        if args.save_aligned_ply is not None:
            aligned_all_local = ground_alignment.apply_transform_matrix(points_zup, T)
            _save_aligned_ply(args.save_aligned_ply, aligned_all_local, colors_uint8)
        if args.exit_on_save:
            raise SystemExit(0)

    print("[auto_ground_alignment_geocalib] Open the Viser viewer URL printed above. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(tyro.cli(AutoGroundAlignArgs))
