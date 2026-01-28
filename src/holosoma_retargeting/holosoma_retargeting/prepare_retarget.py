"""
Prepare human motion input for Holosoma retargeting from SMPL-X reconstruction results.

This module converts per-frame SMPL-X parameters (pose/betas/transl) into a world-frame
joint position sequence, applies coordinate conversion (OpenCV -> Z-up) and an optional
scene alignment transform (e.g. from `ground_alignment.py`), then exports a tensor in an
InterMimic-compatible layout (so it can be consumed by the existing `smplh` loader).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import smplx
import torch
from tqdm import tqdm

from holosoma_retargeting.config_types.data_type import MOCAP_DEMO_JOINTS, SMPLH_DEMO_JOINTS, SMPLX_DEMO_JOINTS


# Copied from the legacy script to keep a stable name->index mapping for SMPL-X joints.
# This list is used only to remap a subset of joints to the 52-joint SMPLH demo convention.
SMPLX_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",
    "left_mouth_4",
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]


NAME_MAP = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip",
    "L_Knee": "left_knee",
    "L_Ankle": "left_ankle",
    "L_Toe": "left_foot",  # no explicit toe in this joint set; use foot
    "R_Hip": "right_hip",
    "R_Knee": "right_knee",
    "R_Ankle": "right_ankle",
    "R_Toe": "right_foot",  # no explicit toe in this joint set; use foot
    "Torso": "spine1",
    "Spine": "spine2",
    "Chest": "spine3",
    "Neck": "neck",
    "Head": "head",
    "L_Thorax": "left_collar",
    "L_Shoulder": "left_shoulder",
    "L_Elbow": "left_elbow",
    "L_Wrist": "left_wrist",
    "L_Index1": "left_index1",
    "L_Index2": "left_index2",
    "L_Index3": "left_index3",
    "L_Middle1": "left_middle1",
    "L_Middle2": "left_middle2",
    "L_Middle3": "left_middle3",
    "L_Pinky1": "left_pinky1",
    "L_Pinky2": "left_pinky2",
    "L_Pinky3": "left_pinky3",
    "L_Ring1": "left_ring1",
    "L_Ring2": "left_ring2",
    "L_Ring3": "left_ring3",
    "L_Thumb1": "left_thumb1",
    "L_Thumb2": "left_thumb2",
    "L_Thumb3": "left_thumb3",
    "R_Thorax": "right_collar",
    "R_Shoulder": "right_shoulder",
    "R_Elbow": "right_elbow",
    "R_Wrist": "right_wrist",
    "R_Index1": "right_index1",
    "R_Index2": "right_index2",
    "R_Index3": "right_index3",
    "R_Middle1": "right_middle1",
    "R_Middle2": "right_middle2",
    "R_Middle3": "right_middle3",
    "R_Pinky1": "right_pinky1",
    "R_Pinky2": "right_pinky2",
    "R_Pinky3": "right_pinky3",
    "R_Ring1": "right_ring1",
    "R_Ring2": "right_ring2",
    "R_Ring3": "right_ring3",
    "R_Thumb1": "right_thumb1",
    "R_Thumb2": "right_thumb2",
    "R_Thumb3": "right_thumb3",
}

MOCAP_TO_SMPLX_NAME_MAP = {
    "Hips": "pelvis",

    "Spine": "spine1",
    "Spine1": "spine3",

    "Neck": "neck",
    "Head": "head",

    "LeftShoulder": "left_collar",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",

    "LeftHandThumb1": "left_thumb1",
    "LeftHandThumb2": "left_thumb2",
    "LeftHandThumb3": "left_thumb3",

    "LeftHandIndex1": "left_index1",
    "LeftHandIndex2": "left_index2",
    "LeftHandIndex3": "left_index3",

    "LeftHandMiddle1": "left_middle1",
    "LeftHandMiddle2": "left_middle2",
    "LeftHandMiddle3": "left_middle3",

    "LeftHandRing1": "left_ring1",
    "LeftHandRing2": "left_ring2",
    "LeftHandRing3": "left_ring3",

    "LeftHandPinky1": "left_pinky1",
    "LeftHandPinky2": "left_pinky2",
    "LeftHandPinky3": "left_pinky3",

    "RightShoulder": "right_collar",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",

    "RightHandThumb1": "right_thumb1",
    "RightHandThumb2": "right_thumb2",
    "RightHandThumb3": "right_thumb3",

    "RightHandIndex1": "right_index1",
    "RightHandIndex2": "right_index2",
    "RightHandIndex3": "right_index3",

    "RightHandMiddle1": "right_middle1",
    "RightHandMiddle2": "right_middle2",
    "RightHandMiddle3": "right_middle3",

    "RightHandRing1": "right_ring1",
    "RightHandRing2": "right_ring2",
    "RightHandRing3": "right_ring3",

    "RightHandPinky1": "right_pinky1",
    "RightHandPinky2": "right_pinky2",
    "RightHandPinky3": "right_pinky3",

    "LeftUpLeg": "left_hip",
    "LeftLeg": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToeBase": "left_foot",

    "RightUpLeg": "right_hip",
    "RightLeg": "right_knee",
    "RightFoot": "right_ankle",
    "RightToeBase": "right_foot",

    "LeftFootMod": "left_heel",
    "RightFootMod": "right_heel",
}


def opencv_to_z_up(xyz: np.ndarray) -> np.ndarray:
    """OpenCV (x right, y down, z forward) -> Z-up (x forward, y left, z up)."""
    out = xyz.copy()
    out[..., 0] = xyz[..., 0]
    out[..., 1] = xyz[..., 2]
    out[..., 2] = -xyz[..., 1]
    return out


def apply_transform_matrix(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points (row-major)."""
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    pts_h = np.concatenate([points, ones], axis=1)
    return (pts_h @ T.T)[:, :3]


def transform_cam_to_world(points_cam: np.ndarray, camera_pose_c2w: np.ndarray, *, scale_translation: float = 1.0) -> np.ndarray:
    """
    Transform points from camera frame to world frame.

    Note: by default we only scale the camera translation (legacy behavior) to match
    a recovered scene scale; point coordinates are left untouched.
    """
    R_cw = camera_pose_c2w[:3, :3]
    t_cw = camera_pose_c2w[:3, 3] * float(scale_translation)
    return (R_cw @ points_cam.T).T + t_cw


def _scale_factor_path(scale_factors_path: Path, frame_idx: int) -> Path:
    return scale_factors_path / f"{frame_idx:04d}_scale_factor.txt"


def load_scale_factor_for_frame(scale_factors_path: str | Path, frame_idx: int) -> float:
    """Load a scale factor for a specific frame from `*_scale_factor.txt`."""
    scale_path = _scale_factor_path(Path(scale_factors_path), frame_idx)
    if not scale_path.exists():
        raise FileNotFoundError(f"Scale factor file not found for frame {frame_idx}: {scale_path}")
    with open(scale_path, "r") as f:
        return float(f.read().strip())


def load_scale_factor_average(scale_factors_path: str | Path, *, max_frames: int = 10000) -> float:
    """Compute the average scale factor across available `*_scale_factor.txt` files."""
    scale_dir = Path(scale_factors_path)
    values: list[float] = []
    for frame_idx in range(max_frames):
        p = _scale_factor_path(scale_dir, frame_idx)
        if not p.exists():
            continue
        try:
            with open(p, "r") as f:
                values.append(float(f.read().strip()))
        except Exception:
            continue
    if not values:
        raise FileNotFoundError(f"No valid scale factor files found under: {scale_dir}")
    return float(np.mean(values))


def _smplx_index_map_to_smplh52() -> np.ndarray:
    index_map: list[int] = []
    for demo_name in SMPLH_DEMO_JOINTS:
        smplx_name = NAME_MAP[demo_name]
        index_map.append(SMPLX_JOINT_NAMES.index(smplx_name))
    return np.asarray(index_map, dtype=np.int64)


_SMPLX_TO_SMPLH52 = _smplx_index_map_to_smplh52()


def remap_smplx_to_smplh52(joints_smplx: np.ndarray) -> np.ndarray:
    """
    Args:
        joints_smplx: (..., J, 3) array using SMPL-X joint order defined by `SMPLX_JOINT_NAMES`.
    Returns:
        (..., 52, 3) array in Holosoma SMPLH demo-joints convention.
    """
    if joints_smplx.shape[-2] != len(SMPLX_JOINT_NAMES):
        raise ValueError(
            f"Unexpected SMPL-X joint count: {joints_smplx.shape[-2]} (expected {len(SMPLX_JOINT_NAMES)})."
        )
    return joints_smplx[..., _SMPLX_TO_SMPLH52, :]


_SMPLX_DEMO_NAME_MAP = {
    "Pelvis": "pelvis",
    "L_Hip": "left_hip",
    "R_Hip": "right_hip",
    "Spine1": "spine1",
    "L_Knee": "left_knee",
    "R_Knee": "right_knee",
    "Spine2": "spine2",
    "L_Ankle": "left_ankle",
    "R_Ankle": "right_ankle",
    "Spine3": "spine3",
    "L_Foot": "left_foot",
    "R_Foot": "right_foot",
    "Neck": "neck",
    "L_Collar": "left_collar",
    "R_Collar": "right_collar",
    "Head": "head",
    "L_Shoulder": "left_shoulder",
    "R_Shoulder": "right_shoulder",
    "L_Elbow": "left_elbow",
    "R_Elbow": "right_elbow",
    "L_Wrist": "left_wrist",
    "R_Wrist": "right_wrist",
}


def remap_smplx_to_smplx_demo(joints_smplx: np.ndarray) -> np.ndarray:
    """
    Args:
        joints_smplx: (..., J, 3) array using SMPL-X joint order defined by `SMPLX_JOINT_NAMES`.
    Returns:
        (..., 22, 3) array in SMPLX_DEMO_JOINTS order.
    """
    if joints_smplx.shape[-2] != len(SMPLX_JOINT_NAMES):
        raise ValueError(
            f"Unexpected SMPL-X joint count: {joints_smplx.shape[-2]} (expected {len(SMPLX_JOINT_NAMES)})."
        )
    indices: list[int] = []
    for demo_name in SMPLX_DEMO_JOINTS:
        smplx_name = _SMPLX_DEMO_NAME_MAP[demo_name]
        indices.append(SMPLX_JOINT_NAMES.index(smplx_name))
    return joints_smplx[..., indices, :]


def _smplx_name_to_index() -> dict[str, int]:
    return {name: idx for idx, name in enumerate(SMPLX_JOINT_NAMES)}


_SMPLX_NAME_TO_INDEX = _smplx_name_to_index()


def remap_smplx_to_mocap(joints_smplx: np.ndarray) -> np.ndarray:
    """
    Remap SMPL-X joints to MOCAP_DEMO_JOINTS order.

    Args:
        joints_smplx: (..., J, 3) array in SMPLX_JOINT_NAMES order.
    Returns:
        (..., len(MOCAP_DEMO_JOINTS), 3) array in MOCAP_DEMO_JOINTS order.
    """
    if joints_smplx.shape[-2] != len(SMPLX_JOINT_NAMES):
        raise ValueError(
            f"Unexpected SMPL-X joint count: {joints_smplx.shape[-2]} (expected {len(SMPLX_JOINT_NAMES)})."
        )

    missing = [j for j in MOCAP_DEMO_JOINTS if j not in MOCAP_TO_SMPLX_NAME_MAP]
    if missing:
        raise ValueError(f"MOCAP_TO_SMPLX_NAME_MAP missing mocap joints: {missing}")

    unknown_smplx = sorted({v for v in MOCAP_TO_SMPLX_NAME_MAP.values() if v not in _SMPLX_NAME_TO_INDEX})
    if unknown_smplx:
        raise ValueError(f"MOCAP_TO_SMPLX_NAME_MAP has unknown SMPL-X joint names: {unknown_smplx}")

    out = [joints_smplx[..., _SMPLX_NAME_TO_INDEX[MOCAP_TO_SMPLX_NAME_MAP[j]], :] for j in MOCAP_DEMO_JOINTS]
    return np.stack(out, axis=-2)


def compute_smplx_joints(
    video_results: list[dict],
    *,
    smpl_model_path: str,
    device: str = "cpu",
    gender: str = "male",
    use_face_contour: bool = True,
) -> list[dict]:
    """Regenerate SMPL-X joints from per-frame SMPL-X parameters."""
    smplx_model = smplx.create(
        model_path=str(smpl_model_path),
        model_type="smplx",
        gender=gender,
        use_face_contour=use_face_contour,
    ).to(device)

    new_video_results: list[dict] = []
    for frame_data in tqdm(video_results, desc="SMPL-X joints"):
        pose = frame_data["pose"]
        betas = frame_data["betas"]
        transl = frame_data["transl"]

        pose_tensor = torch.tensor(pose, dtype=torch.float32, device=device).unsqueeze(0)
        betas_tensor = torch.tensor(betas, dtype=torch.float32, device=device).unsqueeze(0)
        transl_tensor = torch.tensor(transl, dtype=torch.float32, device=device).unsqueeze(0)

        global_orient = pose_tensor[:, :3]
        body_pose = pose_tensor[:, 3:66]
        jaw_pose = pose_tensor[:, 66:69]
        leye_pose = pose_tensor[:, 69:72]
        reye_pose = pose_tensor[:, 72:75]

        with torch.no_grad():
            out = smplx_model(
                global_orient=global_orient,
                body_pose=body_pose,
                jaw_pose=jaw_pose,
                leye_pose=leye_pose,
                reye_pose=reye_pose,
                betas=betas_tensor,
                transl=transl_tensor,
                return_verts=False,
            )
            joints = out.joints[0].detach().cpu().numpy()

        new_frame_data = dict(frame_data)
        new_frame_data["joints"] = joints
        new_video_results.append(new_frame_data)

    return new_video_results


def convert_smplx_results_to_intermimic(
    video_results: list[dict],
    *,
    scale_factors_path: str | Path | None,
    scale_mode: Literal["constant", "per_frame", "average"] = "constant",
    constant_scale_factor: float = 1.0,
    transform_matrix: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Convert SMPL-X results (with per-frame `camera_pose` and computed `joints`) to InterMimic layout.

    Output:
        Tensor of shape (T, 591) with:
          - human joints written to [162 : 162 + 52*3]
          - object pose written to [318 : 325] as [x,y,z,qx,qy,qz,qw] (legacy InterMimic order)
    """
    if transform_matrix is None:
        T_align = np.eye(4, dtype=np.float64)
    else:
        T_align = np.asarray(transform_matrix, dtype=np.float64)
        if T_align.shape != (4, 4):
            raise ValueError(f"transform_matrix must be 4x4, got {T_align.shape}")

    if scale_mode in {"per_frame", "average"} and scale_factors_path is None:
        raise ValueError("scale_factors_path is required for scale_mode='per_frame' or 'average'")

    if scale_mode == "average":
        avg = load_scale_factor_average(scale_factors_path)
    else:
        avg = None

    human_joints_52_list: list[np.ndarray] = []
    for frame_idx, frame in enumerate(video_results):
        if "camera_pose" not in frame:
            raise KeyError("Missing 'camera_pose' in frame data")
        if "joints" not in frame:
            raise KeyError("Missing 'joints' in frame data (run compute_smplx_joints first)")

        camera_pose = np.asarray(frame["camera_pose"], dtype=np.float64)
        joints_cam = np.asarray(frame["joints"], dtype=np.float64)

        if scale_mode == "constant":
            s = float(constant_scale_factor)
        elif scale_mode == "average":
            s = float(avg)
        else:  # per_frame
            s = float(load_scale_factor_for_frame(scale_factors_path, frame_idx))

        joints_world = transform_cam_to_world(joints_cam, camera_pose, scale_translation=s)
        joints_world = opencv_to_z_up(joints_world)
        joints_world = apply_transform_matrix(joints_world, T_align)

        joints_52 = remap_smplx_to_smplh52(joints_world[None, ...])[0]
        human_joints_52_list.append(joints_52.astype(np.float32))

    human_joints = np.stack(human_joints_52_list, axis=0)  # (T, 52, 3)
    human_joints_flat = human_joints.reshape(human_joints.shape[0], -1)  # (T, 156)

    out = torch.zeros(human_joints.shape[0], 591, dtype=torch.float32)
    out[:, 162 : 162 + 52 * 3] = torch.from_numpy(human_joints_flat)

    # Robot-only: default object pose in InterMimic order [x,y,z,qx,qy,qz,qw].
    out[:, 318:325] = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32).unsqueeze(0).repeat(
        human_joints.shape[0], 1
    )

    return out


def convert_smplx_results_to_mocap(
    video_results: list[dict],
    *,
    scale_factors_path: str | Path | None,
    scale_mode: Literal["constant", "per_frame", "average"] = "constant",
    constant_scale_factor: float = 1.0,
    transform_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert SMPL-X results (with per-frame `camera_pose` and computed `joints`) to MOCAP layout.

    Output:
        Array of shape (T, len(MOCAP_DEMO_JOINTS), 3) in MOCAP_DEMO_JOINTS order.
    """
    if transform_matrix is None:
        T_align = np.eye(4, dtype=np.float64)
    else:
        T_align = np.asarray(transform_matrix, dtype=np.float64)
        if T_align.shape != (4, 4):
            raise ValueError(f"transform_matrix must be 4x4, got {T_align.shape}")

    if scale_mode in {"per_frame", "average"} and scale_factors_path is None:
        raise ValueError("scale_factors_path is required for scale_mode='per_frame' or 'average'")

    if scale_mode == "average":
        avg = load_scale_factor_average(scale_factors_path)
    else:
        avg = None

    mocap_frames: list[np.ndarray] = []
    for frame_idx, frame in enumerate(video_results):
        if "camera_pose" not in frame:
            raise KeyError("Missing 'camera_pose' in frame data")
        if "joints" not in frame:
            raise KeyError("Missing 'joints' in frame data (run compute_smplx_joints first)")

        camera_pose = np.asarray(frame["camera_pose"], dtype=np.float64)
        joints_cam = np.asarray(frame["joints"], dtype=np.float64)

        if scale_mode == "constant":
            s = float(constant_scale_factor)
        elif scale_mode == "average":
            s = float(avg)
        else:  # per_frame
            s = float(load_scale_factor_for_frame(scale_factors_path, frame_idx))

        joints_world = transform_cam_to_world(joints_cam, camera_pose, scale_translation=s)
        joints_world = opencv_to_z_up(joints_world)
        joints_world = apply_transform_matrix(joints_world, T_align)

        joints_mocap = remap_smplx_to_mocap(joints_world[None, ...])[0]
        mocap_frames.append(joints_mocap.astype(np.float32))

    return np.stack(mocap_frames, axis=0)


def convert_smplx_results_to_smplx_npz(
    video_results: list[dict],
    *,
    scale_factors_path: str | Path | None,
    scale_mode: Literal["constant", "per_frame", "average"] = "constant",
    constant_scale_factor: float = 1.0,
    transform_matrix: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """
    Convert SMPL-X results to SMPLX_DEMO_JOINTS in world Z-up coordinates.

    Returns:
        dict with keys:
          - global_joint_positions: (T, 22, 3)
          - height: scalar height estimate (meters)
    """
    if transform_matrix is None:
        T_align = np.eye(4, dtype=np.float64)
    else:
        T_align = np.asarray(transform_matrix, dtype=np.float64)
        if T_align.shape != (4, 4):
            raise ValueError(f"transform_matrix must be 4x4, got {T_align.shape}")

    if scale_mode in {"per_frame", "average"} and scale_factors_path is None:
        raise ValueError("scale_factors_path is required for scale_mode='per_frame' or 'average'")

    if scale_mode == "average":
        avg = load_scale_factor_average(scale_factors_path)
    else:
        avg = None

    joints_demo_list: list[np.ndarray] = []
    for frame_idx, frame in enumerate(video_results):
        if "camera_pose" not in frame:
            raise KeyError("Missing 'camera_pose' in frame data")
        if "joints" not in frame:
            raise KeyError("Missing 'joints' in frame data (run compute_smplx_joints first)")

        camera_pose = np.asarray(frame["camera_pose"], dtype=np.float64)
        joints_cam = np.asarray(frame["joints"], dtype=np.float64)

        if scale_mode == "constant":
            s = float(constant_scale_factor)
        elif scale_mode == "average":
            s = float(avg)
        else:  # per_frame
            s = float(load_scale_factor_for_frame(scale_factors_path, frame_idx))

        joints_world = transform_cam_to_world(joints_cam, camera_pose, scale_translation=s)
        joints_world = opencv_to_z_up(joints_world)
        joints_world = apply_transform_matrix(joints_world, T_align)

        joints_demo = remap_smplx_to_smplx_demo(joints_world[None, ...])[0]
        joints_demo_list.append(joints_demo.astype(np.float32))

    joints_demo = np.stack(joints_demo_list, axis=0)
    z = joints_demo[..., 2]
    height = float(z.max() - z.min())
    return {
        "global_joint_positions": joints_demo,
        "height": np.asarray(height, dtype=np.float32),
    }


@dataclass(frozen=True)
class PrepareRetargetConfig:
    input_all_results_path: Path
    output_pt_path: Path
    smpl_model_path: Path
    device: str = "cpu"
    gender: str = "male"
    use_face_contour: bool = True

    # Optional alignment and scaling
    transform_matrix: Optional[np.ndarray] = None
    scale_factors_path: Optional[Path] = None
    scale_mode: Literal["constant", "per_frame", "average"] = "constant"
    constant_scale_factor: float = 1.0


def main(config: PrepareRetargetConfig) -> None:
    video_results = torch.load(str(config.input_all_results_path), map_location="cpu", weights_only=False)

    new_video_results = compute_smplx_joints(
        video_results,
        smpl_model_path=str(config.smpl_model_path),
        device=config.device,
        gender=config.gender,
        use_face_contour=config.use_face_contour,
    )

    converted = convert_smplx_results_to_intermimic(
        new_video_results,
        transform_matrix=config.transform_matrix,
        scale_factors_path=str(config.scale_factors_path) if config.scale_factors_path is not None else None,
        scale_mode=config.scale_mode,
        constant_scale_factor=config.constant_scale_factor,
    )

    config.output_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, str(config.output_pt_path))


if __name__ == "__main__":
    raise SystemExit("Use prepare_retarget.main(PrepareRetargetConfig(...)) from your pipeline.")
