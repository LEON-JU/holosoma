"""
Unified pipeline runner (new-version workspace).

Runs the end-to-end flow:
1) Interactive ground alignment (optional): produces a stable transform JSON artifact.
2) Prepare retarget input from SMPL-X results: produces an InterMimic-style `.pt`.
3) Run Holosoma retargeting on the prepared data.
4) Save a run summary JSON for downstream tools.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tyro
import torch

from holosoma_retargeting import ground_alignment
from holosoma_retargeting import prepare_retarget
from holosoma_retargeting.config import (
    PipelineArgs,
    ensure_run_dirs,
    get_scale_factor,
    get_sequence_paths,
)
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.examples import robot_retarget
from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS
from holosoma_retargeting.ply2scene.scene_xml import build_robot_scene_xml, create_scaled_scene_urdf


def _load_transform_json(path: Path) -> np.ndarray:
    with open(path, "r") as f:
        data = json.load(f)
    T = np.asarray(data["transform_matrix"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"transform_matrix in {path} must be 4x4, got {T.shape}")
    return T


def _save_transform_json(path: Path, T: np.ndarray, *, seq: str, robot: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "transform_matrix": np.asarray(T, dtype=np.float64).round(8).tolist(),
        "timestamp": datetime.now().isoformat(),
        "seq": seq,
        "robot": robot,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _resolve_ground_alignment_mode(args: PipelineArgs) -> str:
    return str(args.ground_alignment_mode)


def _compute_vertical_bias_z_min_from_intermimic_pt(
    pt_path: Path,
    *,
    toe_joint_names: tuple[str, str] = ("L_Toe", "R_Toe"),
    mat_height: float = 0.1,
) -> float:
    """Match `src/utils.py:preprocess_motion_data` z-min computation (before scaling)."""
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not hasattr(data, "shape") or len(data.shape) != 2 or data.shape[1] < 162 + 52 * 3:
        raise ValueError(f"Unexpected InterMimic tensor shape in {pt_path}: {getattr(data, 'shape', None)}")
    joints_flat = data[:, 162 : 162 + 52 * 3].detach().cpu().numpy()
    joints = joints_flat.reshape(-1, 52, 3)
    toe_indices = [SMPLH_DEMO_JOINTS.index(toe_joint_names[0]), SMPLH_DEMO_JOINTS.index(toe_joint_names[1])]
    z_min = float(joints[:, toe_indices, 2].min())
    if z_min >= mat_height:
        z_min -= mat_height
    return z_min


def _save_contact_logits(pt_path: Path, out_path: Path) -> None:
    results = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    logits: list[np.ndarray] = []
    for frame_data in results:
        if isinstance(frame_data, dict) and frame_data.get("static_conf_logits") is not None:
            conf = np.asarray(frame_data["static_conf_logits"], dtype=np.float32).reshape(-1)
            logits.append(conf)
        else:
            logits.append(np.zeros(6, dtype=np.float32))
    if not logits:
        raise ValueError(f"No frames found when extracting contact logits from {pt_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        static_conf_logits=np.stack(logits, axis=0),
        smplx_joint_ids=np.asarray([7, 10, 8, 11, 20, 21], dtype=np.int64),
    )


def _write_pipeline_config(
    *,
    path: Path,
    args: PipelineArgs,
    paths,
    scale_factor: float,
    stage: str,
    retarget_mode: str,
    vertical_bias_z_min_human_m: float | None = None,
    vertical_bias_z_min_robot_m: float | None = None,
) -> None:
    payload = {
        "seq": args.seq,
        "robot": args.robot,
        "human_height_m": args.human_height_m,
        "scale_factor": float(scale_factor),
        "retarget_mode": retarget_mode,
        "stage": stage,
        "transform_path": str(paths.ground_transform_json),
        "prepared_pt": str(paths.prepared_pt_path),
        "retarget_save_dir": str(paths.retarget_save_dir),
        "paths": {k: str(v) for k, v in asdict(paths).items()},
        "vertical_bias_z_min_human_m": vertical_bias_z_min_human_m,
        "vertical_bias_z_min_robot_m": vertical_bias_z_min_robot_m,
        "timestamp": datetime.now().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main(args: PipelineArgs) -> None:
    paths = get_sequence_paths(seq=args.seq, robot=args.robot, manual=args.manual)
    ensure_run_dirs(paths)

    scale_factor = get_scale_factor(args.robot, args.human_height_m)

    # 1) Ground alignment
    mode = _resolve_ground_alignment_mode(args)
    if mode == "manual":
        ga_cfg = {
            "smpl_model_path": str(args.manual.smpl_model_path),
            "human_mesh_path": str(paths.all_results_video),
            "scale_factors_path": str(paths.depth_recovered),
            "scene_ply_path": str(paths.fused_scene_ply),
            "default_save_path": str(paths.aligned_scene_ply),
            "data_dir": str(paths.predicted_dir),
            "depth_dir": str(paths.depth_recovered),
            "smplx_results_dir": str(paths.results_dir),
            "use_rgbd_scene": True,
            "transform_output_path": str(paths.ground_transform_json),
            "exit_process_on_save": False,
        }

        T_align = ground_alignment.main(ga_cfg)
        _save_transform_json(paths.ground_transform_json, T_align, seq=args.seq, robot=args.robot)
    elif mode == "cached":
        if not paths.ground_transform_json.exists():
            raise FileNotFoundError(f"ground_alignment_mode='cached' but transform not found: {paths.ground_transform_json}")
        T_align = _load_transform_json(paths.ground_transform_json)
    elif mode == "geocalib":
        from holosoma_retargeting.postprocess.geocalib_ground_alignment import (
            GeoCalibGroundAlignmentConfig,
            compute_geocalib_ground_alignment,
        )

        cfg = GeoCalibGroundAlignmentConfig(
            predicted_dir=paths.predicted_dir,
            depth_dir=paths.depth_recovered,
            results_dir=paths.results_dir,
            fused_scene_ply=paths.fused_scene_ply,
            use_rgbd_scene=bool(args.geocalib_alignment.use_rgbd_scene),
            num_geocalib_frames=int(args.geocalib_alignment.num_geocalib_frames),
            geocalib_frame_stride=int(args.geocalib_alignment.geocalib_frame_stride),
            geocalib_frame_indices=args.geocalib_alignment.geocalib_frame_indices,
            geocalib_weights=str(args.geocalib_alignment.geocalib_weights),
            geocalib_camera_y_up=bool(args.geocalib_alignment.geocalib_camera_y_up),
            geocalib_device=args.geocalib_alignment.geocalib_device,
            geocalib_angle_outlier_deg=float(args.geocalib_alignment.geocalib_angle_outlier_deg),
            max_points=int(args.geocalib_alignment.max_points),
            voxel_size=float(args.geocalib_alignment.voxel_size),
            plane_distance_threshold=float(args.geocalib_alignment.plane_distance_threshold),
            plane_ransac_n=int(args.geocalib_alignment.plane_ransac_n),
            plane_num_iterations=int(args.geocalib_alignment.plane_num_iterations),
            plane_angle_deg=float(args.geocalib_alignment.plane_angle_deg),
            plane_max_candidates=int(args.geocalib_alignment.plane_max_candidates),
            recenter_xy=bool(args.geocalib_alignment.recenter_xy),
        )
        result = compute_geocalib_ground_alignment(cfg, return_points=False, debug=bool(args.geocalib_alignment.debug))
        T_align = result.T_align
        _save_transform_json(paths.ground_transform_json, T_align, seq=args.seq, robot=args.robot)
    else:
        raise ValueError(f"Unknown ground_alignment_mode: {mode}")

    # If we're only preparing artifacts (transform + scale_factor), write pipeline_config.json now and exit.
    if str(getattr(args, "stage", "full")) == "prepare":
        _write_pipeline_config(
            path=paths.pipeline_config_json,
            args=args,
            paths=paths,
            scale_factor=scale_factor,
            stage="prepare",
            retarget_mode=str(args.retarget_mode),
        )
        print(f"[pipeline] Prepared artifacts for {args.seq}/{args.robot}")
        print(f"[pipeline] Transform: {paths.ground_transform_json}")
        print(f"[pipeline] Config: {paths.pipeline_config_json}")
        return

    # 2) Prepare retarget data
    if args.retarget_mode == "robot_only":
        prep_cfg = prepare_retarget.PrepareRetargetConfig(
            input_all_results_path=paths.all_results_video,
            output_pt_path=paths.prepared_pt_path,
            smpl_model_path=args.manual.smpl_model_path,
            device="cpu",
            transform_matrix=T_align,
            scale_factors_path=paths.depth_recovered,
            # Scene scale recovery (NOT robot/human scaling): use recovered depth scale factors.
            scale_mode="average",
            constant_scale_factor=1.0,
        )
        prepare_retarget.main(prep_cfg)
    elif args.retarget_mode == "climbing_scene":
        video_results = torch.load(str(paths.all_results_video), map_location="cpu", weights_only=False)
        new_video_results = prepare_retarget.compute_smplx_joints(
            video_results,
            smpl_model_path=str(args.manual.smpl_model_path),
            device="cpu",
            gender="male",
            use_face_contour=True,
        )
        smplx_npz = prepare_retarget.convert_smplx_results_to_smplx_npz(
            new_video_results,
            transform_matrix=T_align,
            scale_factors_path=str(paths.depth_recovered),
            scale_mode="average",
            constant_scale_factor=1.0,
        )
        npz_path = paths.prepared_data_dir / f"{args.seq}.npz"
        np.savez(str(npz_path), **smplx_npz)
    else:
        raise ValueError(f"Unknown retarget_mode: {args.retarget_mode}")

    contact_npz_path = paths.prepared_data_dir / f"{args.seq}_contact.npz"
    _save_contact_logits(paths.all_results_video, contact_npz_path)

    if args.retarget_mode == "robot_only":
        # This is the vertical bias printed by `src/utils.py:preprocess_motion_data` (before scaling).
        vertical_bias_z_min_human_m = _compute_vertical_bias_z_min_from_intermimic_pt(paths.prepared_pt_path)
        vertical_bias_z_min_robot_m = vertical_bias_z_min_human_m * scale_factor
    else:
        vertical_bias_z_min_human_m = None
        vertical_bias_z_min_robot_m = None

    # 3) Retarget
    if args.retarget_mode == "robot_only":
        rt_cfg = RetargetingConfig(
            task_type="robot_only",
            robot=args.robot,
            data_format="smplh",
            task_name=f"{args.seq}",
            data_path=paths.prepared_data_dir,
            save_dir=paths.retarget_save_dir,
            augmentation=False,
            custom_scale_factor=scale_factor,
            retargeter=RetargeterConfig(visualize=True, debug=True),
        )
    else:
        task_dir = paths.prepared_data_dir / args.seq
        scene_pkg_dir = paths.run_dir / "artifacts" / "ply2scene" / "scene"
        if not scene_pkg_dir.exists():
            raise FileNotFoundError(f"Scene package not found: {scene_pkg_dir} (run ply2scene.convert first)")

        robot_config = RobotConfig(robot_type=args.robot)
        robot_urdf_path = Path(args.robot_urdf_file) if args.robot_urdf_file is not None else Path(
            robot_config.ROBOT_URDF_FILE
        )
        if not robot_urdf_path.is_absolute():
            package_root = Path(__file__).resolve().parents[1]
            robot_urdf_path = package_root / robot_urdf_path

        robot_xml_path = robot_urdf_path.with_suffix(".xml")
        if not robot_xml_path.exists():
            raise FileNotFoundError(f"Robot XML not found: {robot_xml_path}")

        # Build (or refresh) robot+scene MJCF inside the ply2scene output directory.
        build_robot_scene_xml(
            robot_xml_path,
            scene_pkg_dir,
            scene_pkg_dir,
            scale=scale_factor,
            output_name="robot_scene.xml",
            disable_plane_ground=False,
        )

        scene_urdf_src = scene_pkg_dir / "scene.urdf"
        if not scene_urdf_src.exists():
            raise FileNotFoundError(f"Scene URDF not found: {scene_urdf_src}")
        # Ensure the URDF mesh scale matches the pipeline scale_factor (needed for Viser/yourdfpy rendering).
        create_scaled_scene_urdf(scene_urdf_src, scale_factor, output_path=scene_urdf_src)
        expected_scale = f'scale="{scale_factor} {scale_factor} {scale_factor}"'
        if expected_scale not in scene_urdf_src.read_text():
            raise RuntimeError(
                f"Failed to update scene URDF mesh scale to {scale_factor}. "
                f"Expected to find {expected_scale} in {scene_urdf_src}."
            )

        rt_cfg = RetargetingConfig(
            task_type="climbing",
            robot=args.robot,
            data_format="smplx",
            task_name=f"{args.seq}",
            data_path=paths.prepared_data_dir,
            save_dir=paths.retarget_save_dir,
            augmentation=False,
            custom_scale_factor=scale_factor,
            retargeter=RetargeterConfig(visualize=True, debug=True),
        )
        # NOTE: motion is read from `data_path/task_name` (task_dir), but scene assets are read from `object_dir`.
        rt_cfg.task_config = replace(rt_cfg.task_config, object_name="scene", object_dir=scene_pkg_dir)
        rt_cfg.robot_config = replace(rt_cfg.robot_config, robot_urdf_file=str(robot_urdf_path))

    robot_retarget.main(rt_cfg)

    # 4) Save run summary for downstream tools
    _write_pipeline_config(
        path=paths.pipeline_config_json,
        args=args,
        paths=paths,
        scale_factor=scale_factor,
        stage="full",
        retarget_mode=str(args.retarget_mode),
        vertical_bias_z_min_human_m=vertical_bias_z_min_human_m,
        vertical_bias_z_min_robot_m=vertical_bias_z_min_robot_m,
    )


if __name__ == "__main__":
    main(tyro.cli(PipelineArgs))
