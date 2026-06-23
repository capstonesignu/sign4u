"""
preprocess.py

AI Hub 수어 데이터(.npz)에서 키포인트를 로드하고 temporal interpolation으로
시퀀스 길이를 정규화한다. npz는 이미 추출된 keypoint이므로 MediaPipe 불필요.

Landmark 구조 (AI Hub 고정):
    pose       : 9
    left_hand  : 21
    right_hand : 21
    face       : 19
    각 좌표는 (x, y, z) 3축

Feature preset:
    A → pose + hands + face, 3축 → 210 dim
    B → pose + hands + face, 2축 → 140 dim
    C → pose + hands,        2축 → 102 dim
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm


# ────────────────────────────────────────────────────────────
# Feature preset
# ────────────────────────────────────────────────────────────
LANDMARK_COUNTS = {"pose": 7, "left_hand": 21, "right_hand": 21, "face": 19}
_NPZ_POSE_COUNT = 9  # npz에 저장된 pose landmark 수. 앞 7개만 사용 (hip 2개 제외)

FEATURE_PRESETS: Dict[str, Dict] = {
    "A": {"use": ["pose", "left_hand", "right_hand", "face"], "axes": 3},
    "B": {"use": ["pose", "left_hand", "right_hand", "face"], "axes": 2},
    "C": {"use": ["pose", "left_hand", "right_hand"],         "axes": 2},
}

TARGET_LENGTH = 64
LOGGER = logging.getLogger(__name__)


def feature_dim_for(preset: str) -> int:
    """preset 코드 → feature_dim (단일 frame 벡터 차원)."""
    if preset not in FEATURE_PRESETS:
        raise ValueError(f"Unknown feature preset: {preset}. "
                         f"Expected one of {sorted(FEATURE_PRESETS)}")
    cfg = FEATURE_PRESETS[preset]
    n_landmarks = sum(LANDMARK_COUNTS[k] for k in cfg["use"])
    return n_landmarks * cfg["axes"]


# ────────────────────────────────────────────────────────────
# 어깨 기준 좌표 정규화
# ────────────────────────────────────────────────────────────
# 9점 pose 배열에서 각 landmark 위치
# POSE_INDICES = [0(nose), 11(L_sh), 12(R_sh), 13, 14, 15, 16, 23(L_hip), 24(R_hip)]
_SHOULDER_L = 1
_SHOULDER_R = 2
_HIP_L      = 7  # MediaPipe pose landmark 23 = left hip
_HIP_R      = 8  # MediaPipe pose landmark 24 = right hip


def normalize_by_shoulders(frames: np.ndarray) -> np.ndarray:
    """어깨 중점 centering + 어깨 너비 scaling.

    입력: (T, total_landmarks, axes)  — pose가 첫 9개 landmark
    출력: 동일 shape, x/y를 어깨 기준으로 정규화 (axes=3이면 z도 동일 scale)
    """
    l_sh = frames[:, _SHOULDER_L, :2]          # (T, 2)
    r_sh = frames[:, _SHOULDER_R, :2]          # (T, 2)

    mid   = (l_sh + r_sh) / 2.0                # (T, 2)
    width = np.linalg.norm(r_sh - l_sh, axis=-1, keepdims=True)  # (T, 1)
    width = np.maximum(width, 1e-6)

    out = frames.copy()
    out[..., 0] = (frames[..., 0] - mid[:, 0:1]) / width
    out[..., 1] = (frames[..., 1] - mid[:, 1:2]) / width
    if frames.shape[-1] >= 3:
        out[..., 2] = frames[..., 2] / width

    return out


# ────────────────────────────────────────────────────────────
# Camera field-of-view occlusion masking
# ────────────────────────────────────────────────────────────

def apply_camera_occlusion_mask(
    pose9: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> tuple:
    """Simulate camera FOV cutoff at the shoulder-hip midpoint.

    Our demo camera cannot see below the midpoint between shoulder and hip.
    Frames where the wrist (hand landmark 0) y-coordinate is below (> in
    MediaPipe coords where y increases downward) this midpoint are treated
    as out-of-frame: the entire hand frame is zeroed. Zeros are then filled
    by interpolate_missing_keypoints downstream.

    Args:
        pose9:      (T, 9, axes) raw pose — index 7=L_hip, 8=R_hip (raw coords)
        left_hand:  (T, 21, axes) raw hand keypoints
        right_hand: (T, 21, axes) raw hand keypoints

    Returns:
        (left_hand_masked, right_hand_masked)
    """
    shoulder_y = (pose9[:, _SHOULDER_L, 1] + pose9[:, _SHOULDER_R, 1]) / 2.0
    hip_y      = (pose9[:, _HIP_L, 1]      + pose9[:, _HIP_R, 1])      / 2.0

    # Only apply when both hip landmarks are actually detected (non-zero)
    hip_norm   = np.linalg.norm(pose9[:, [_HIP_L, _HIP_R], :2], axis=(1, 2))
    hip_valid  = hip_norm > 1e-4

    # threshold = shoulder-hip midpoint; fall back to 1.0 (no masking) if hip missing
    threshold_y = np.where(hip_valid, (shoulder_y + hip_y) / 2.0, 1.0)

    lh_out = left_hand.copy()
    rh_out = right_hand.copy()
    lh_out[left_hand[:, 0, 1]  > threshold_y] = 0.0
    rh_out[right_hand[:, 0, 1] > threshold_y] = 0.0

    return lh_out, rh_out


# ────────────────────────────────────────────────────────────
# Missing keypoint interpolation
# ────────────────────────────────────────────────────────────
def interpolate_missing_keypoints(arr: np.ndarray,
                                   zero_threshold: float = 1e-4) -> np.ndarray:
    """Linear-interpolate frames where ALL landmarks for a body part are zero.

    AI Hub data has ~3.5% right-hand frames where MediaPipe returns all-zeros
    (detection failure). These must be imputed before normalization to avoid
    spurious features at the shoulder-origin point.

    Args:
        arr: (T, n_landmarks, axes)
        zero_threshold: L2 norm below which a frame is considered missing

    Returns:
        arr with missing frames linearly interpolated from neighbors.
    """
    T = arr.shape[0]
    # Frame is missing when all landmark norms are below threshold
    norms = np.linalg.norm(arr, axis=(1, 2))
    missing = norms < zero_threshold

    if not missing.any():
        return arr

    out = arr.copy()
    # Find valid (non-missing) frame indices
    valid_idx = np.where(~missing)[0]
    if len(valid_idx) == 0:
        return out  # All frames missing — can't interpolate

    for t in np.where(missing)[0]:
        # Find nearest valid frames before and after
        before = valid_idx[valid_idx < t]
        after  = valid_idx[valid_idx > t]
        if len(before) == 0:
            out[t] = out[after[0]]  # Extrapolate from first valid
        elif len(after) == 0:
            out[t] = out[before[-1]]  # Extrapolate from last valid
        else:
            t0, t1 = before[-1], after[0]
            alpha = (t - t0) / (t1 - t0)
            out[t] = (1 - alpha) * arr[t0] + alpha * arr[t1]
    return out


# ────────────────────────────────────────────────────────────
# npz 로더 + temporal interpolation
# ────────────────────────────────────────────────────────────
def load_npz_keypoints(npz_path: str,
                       preset: str,
                       target_length: int = TARGET_LENGTH,
                       camera_occlusion: bool = False) -> np.ndarray:
    """
    역할: 단일 .npz 에서 선택한 요소의 keypoint를 로드 + 평탄화 + 시퀀스 정규화
    입력: npz 경로, preset, target_length
    출력: (target_length, feature_dim) float32

    camera_occlusion=True: shoulder-hip 중점 아래 손 프레임 마스킹 (demo 카메라 시뮬레이션)
    """
    cfg = FEATURE_PRESETS[preset]
    data = np.load(npz_path, allow_pickle=True)

    # 카메라 오클루전: hand 마스킹 전에 pose9 전체 필요
    raw_arrays = {key: data[key] for key in cfg["use"]}
    if camera_occlusion and "pose" in data:
        pose9_raw = data["pose"]  # (T, 9, 3) 원본
        if (pose9_raw.ndim == 3 and pose9_raw.shape[1] == _NPZ_POSE_COUNT
                and "left_hand" in raw_arrays and "right_hand" in raw_arrays):
            raw_arrays["left_hand"], raw_arrays["right_hand"] = apply_camera_occlusion_mask(
                pose9_raw, raw_arrays["left_hand"], raw_arrays["right_hand"]
            )

    parts: List[np.ndarray] = []
    T = None
    for key in cfg["use"]:
        arr = raw_arrays[key]
        # pose는 npz에 9개 저장, 앞 7개만 사용 (hip landmark 제외)
        expected = _NPZ_POSE_COUNT if key == "pose" else LANDMARK_COUNTS[key]
        if arr.ndim != 3 or arr.shape[1] != expected:
            raise ValueError(f"{npz_path} {key} shape mismatch: {arr.shape}")
        if key == "pose":
            arr = arr[:, :LANDMARK_COUNTS["pose"], :]  # (T, 7, 3)
        if cfg["axes"] == 2:
            arr = arr[..., :2]
        if T is None:
            T = arr.shape[0]
        elif arr.shape[0] != T:
            raise ValueError(f"{npz_path} inconsistent T across keys")
        # Interpolate missing frames for hand keypoints (MediaPipe dropout + 카메라 오클루전)
        if key in ("left_hand", "right_hand"):
            arr = interpolate_missing_keypoints(arr)
        parts.append(arr)  # (T, n_landmarks, axes) — 아직 flatten 안 함

    # (T, total_landmarks, axes) → 어깨 정규화 → flatten
    stacked = np.concatenate(parts, axis=1)
    stacked = normalize_by_shoulders(stacked)
    seq = stacked.reshape(T, -1).astype(np.float32)

    return temporal_interpolate(seq, target_length)


def _load_npz_raw(npz_path: str, preset: str, camera_occlusion: bool = False) -> np.ndarray:
    """npz → shoulder-normalized, flattened (T, feature_dim). 시간 정규화 없음.

    camera_occlusion=True: shoulder-hip 중점 아래 손 프레임 마스킹
    """
    cfg = FEATURE_PRESETS[preset]
    data = np.load(npz_path, allow_pickle=True)

    raw_arrays = {key: data[key] for key in cfg["use"]}
    if camera_occlusion and "pose" in data:
        pose9_raw = data["pose"]
        if (pose9_raw.ndim == 3 and pose9_raw.shape[1] == _NPZ_POSE_COUNT
                and "left_hand" in raw_arrays and "right_hand" in raw_arrays):
            raw_arrays["left_hand"], raw_arrays["right_hand"] = apply_camera_occlusion_mask(
                pose9_raw, raw_arrays["left_hand"], raw_arrays["right_hand"]
            )

    parts: List[np.ndarray] = []
    T = None
    for key in cfg["use"]:
        arr = raw_arrays[key]
        expected = _NPZ_POSE_COUNT if key == "pose" else LANDMARK_COUNTS[key]
        if arr.ndim != 3 or arr.shape[1] != expected:
            raise ValueError(f"{npz_path} {key} shape mismatch: {arr.shape}")
        if key == "pose":
            arr = arr[:, :LANDMARK_COUNTS["pose"], :]
        if cfg["axes"] == 2:
            arr = arr[..., :2]
        if T is None:
            T = arr.shape[0]
        if key in ("left_hand", "right_hand"):
            arr = interpolate_missing_keypoints(arr)
        parts.append(arr)
    stacked = np.concatenate(parts, axis=1)
    stacked = normalize_by_shoulders(stacked)
    return stacked.reshape(T, -1).astype(np.float32)


def load_sentence_stream(npz_path: str, preset: str) -> tuple:
    """문장 전체 keypoint 스트림 로드 (temporal interpolation 없음).

    반환: (stream: (T, feature_dim) float32, fps: float)
    segment.py의 segment_X 함수에 직접 전달하는 용도.
    """
    data = np.load(npz_path, allow_pickle=True)
    fps = float(data["fps"])
    stream = _load_npz_raw(npz_path, preset)
    return stream, fps


def load_sentence_segment(npz_path: str,
                          start_sec: float,
                          end_sec: float,
                          preset: str,
                          target_length: int = TARGET_LENGTH) -> np.ndarray:
    """문장 npz에서 [start_sec, end_sec] 구간 세그먼트를 추출.

    입력: 문장 npz 경로, 시작/종료 초, preset, target_length
    출력: (target_length, feature_dim) float32
    """
    data = np.load(npz_path, allow_pickle=True)
    fps = float(data["fps"])
    total_frames = data["pose"].shape[0]

    start_f = max(0, int(round(start_sec * fps)))
    end_f   = min(total_frames, int(round(end_sec * fps)))
    if end_f <= start_f:
        end_f = min(total_frames, start_f + 1)

    seq = _load_npz_raw(npz_path, preset)  # (T_total, D)
    segment = seq[start_f:end_f]           # (T_seg, D)
    return temporal_interpolate(segment, target_length)


def temporal_interpolate(sequence: np.ndarray,
                         target_length: int = TARGET_LENGTH) -> np.ndarray:
    """
    역할: 가변 길이 시퀀스를 target_length 로 보간 정규화
    입력: (T, D)
    출력: (target_length, D) float32
    """
    if sequence.ndim != 2:
        raise ValueError(f"sequence must be 2D, got shape {sequence.shape}")
    T = sequence.shape[0]
    if T == target_length:
        return sequence.astype(np.float32)
    if T == 0:
        raise ValueError("empty sequence")
    if T == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)
    src_idx = np.linspace(0, T - 1, target_length)
    out = np.array([sequence[int(round(i))] for i in src_idx])
    return out.astype(np.float32)


# ────────────────────────────────────────────────────────────
# 데이터셋 일괄 로드
# ────────────────────────────────────────────────────────────
def preprocess_dataset(dataset: Dict[str, Dict[str, List[str]]],
                       preset: str,
                       target_length: int = TARGET_LENGTH
                       ) -> Dict[str, Dict[str, List[np.ndarray]]]:
    """
    역할: {word: {signer: [npz_path, ...]}} → {word: {signer: [np.ndarray(T, D)]}}
    입력: data.load_dataset 결과, feature preset
    출력: 같은 nested 구조, 값이 keypoint ndarray
    주의: npz 파일 자체가 캐시이므로 별도 디스크 캐시 없음.
          로드 실패 시 해당 샘플 skip + 로그.
    """
    out: Dict[str, Dict[str, List[np.ndarray]]] = {}
    total = sum(len(lst) for sm in dataset.values() for lst in sm.values())
    pbar = tqdm(total=total, desc=f"[preprocess preset={preset}]")
    for word, signer_map in dataset.items():
        for signer, paths in signer_map.items():
            for path in paths:
                try:
                    kp = load_npz_keypoints(path, preset, target_length)
                except Exception as e:
                    LOGGER.warning(f"skip {path}: {e}")
                    pbar.update(1)
                    continue
                out.setdefault(word, {}).setdefault(signer, []).append(kp)
                pbar.update(1)
    pbar.close()

    cleaned = {}
    for w, sm in out.items():
        sm2 = {s: lst for s, lst in sm.items() if lst}
        if sm2:
            cleaned[w] = sm2
    return cleaned


# ────────────────────────────────────────────────────────────
# Smoke test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    for p in ("A", "B", "C"):
        print(f"preset {p} → feature_dim = {feature_dim_for(p)}")

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        path = sys.argv[1]
        for p in ("A", "B", "C"):
            kp = load_npz_keypoints(path, preset=p)
            print(f"  {p}: {kp.shape} dtype={kp.dtype}")
