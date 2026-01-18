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
from dataclasses import asdict
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
from holosoma_retargeting.examples import robot_retarget
from holosoma_retargeting.config_types.data_type import SMPLH_DEMO_JOINTS


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


def main(args: PipelineArgs) -> None:
    paths = get_sequence_paths(seq=args.seq, robot=args.robot, manual=args.manual)
    ensure_run_dirs(paths)

    scale_factor = get_scale_factor(args.robot, args.human_height_m)

    # 1) Ground alignment
    if args.run_ground_alignment:
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
    else:
        if not paths.ground_transform_json.exists():
            raise FileNotFoundError(f"run_ground_alignment=False but transform not found: {paths.ground_transform_json}")
        T_align = _load_transform_json(paths.ground_transform_json)

    # 2) Prepare retarget data (InterMimic-style `.pt`)
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

    # This is the vertical bias printed by `src/utils.py:preprocess_motion_data` (before scaling).
    vertical_bias_z_min_human_m = _compute_vertical_bias_z_min_from_intermimic_pt(paths.prepared_pt_path)
    vertical_bias_z_min_robot_m = vertical_bias_z_min_human_m * scale_factor

    # 3) Retarget (robot_only, smplh)
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
    robot_retarget.main(rt_cfg)

    # 4) Save run summary for downstream tools
    summary = {
        "seq": args.seq,
        "robot": args.robot,
        "human_height_m": args.human_height_m,
        "scale_factor": scale_factor,
        "vertical_bias_z_min_human_m": vertical_bias_z_min_human_m,
        "vertical_bias_z_min_robot_m": vertical_bias_z_min_robot_m,
        "paths": {k: str(v) for k, v in asdict(paths).items()},
        "transform_path": str(paths.ground_transform_json),
        "prepared_pt": str(paths.prepared_pt_path),
        "retarget_save_dir": str(paths.retarget_save_dir),
        "timestamp": datetime.now().isoformat(),
    }
    with open(paths.pipeline_config_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main(tyro.cli(PipelineArgs))
