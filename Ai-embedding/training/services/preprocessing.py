import numpy as np

# 실제 feature에 사용하는 landmark 수 (pose: hip 2개 제외)
LANDMARK_COUNTS = {"pose": 7, "left_hand": 21, "right_hand": 21, "face": 19}
LANDMARK_ORDER = ["pose", "left_hand", "right_hand", "face"]

# 프론트엔드가 전송하는 landmark 수 (70개 고정, pose는 9개 포함)
_FRONTEND_COUNTS = {"pose": 9, "left_hand": 21, "right_hand": 21, "face": 19}

FEATURE_PRESETS = {
    "A": {"use": ["pose", "left_hand", "right_hand", "face"], "axes": 3},
    "B": {"use": ["pose", "left_hand", "right_hand", "face"], "axes": 2},
    "C": {"use": ["pose", "left_hand", "right_hand"], "axes": 2},
}

# pose 배열에서 shoulder 위치 (전체 70 landmark 기준)
# POSE_INDICES = [0(nose), 11(L_sh), 12(R_sh), ...]  → overall index 1, 2
_SHOULDER_L = 1
_SHOULDER_R = 2


def _normalize_by_shoulders(frames: np.ndarray) -> np.ndarray:
    """어깨 중점 centering + 어깨 너비 scaling.

    입력: (T, 70, 3)  — 프론트엔드 전체 landmark
    출력: 동일 shape, x/y/z를 어깨 기준으로 정규화
    """
    l_sh = frames[:, _SHOULDER_L, :2]
    r_sh = frames[:, _SHOULDER_R, :2]

    mid   = (l_sh + r_sh) / 2.0
    width = np.linalg.norm(r_sh - l_sh, axis=-1, keepdims=True)
    width = np.maximum(width, 1e-6)

    out = frames.copy()
    out[..., 0] = (frames[..., 0] - mid[:, 0:1]) / width
    out[..., 1] = (frames[..., 1] - mid[:, 1:2]) / width
    out[..., 2] = frames[..., 2] / width

    return out


def feature_dim_for(preset: str) -> int:
    cfg = FEATURE_PRESETS[preset]
    n_landmarks = sum(LANDMARK_COUNTS[k] for k in cfg["use"])
    return n_landmarks * cfg["axes"]


def temporal_interpolate(sequence: np.ndarray, target_length: int) -> np.ndarray:
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


def _interpolate_missing(arr: np.ndarray, zero_threshold: float = 1e-4) -> np.ndarray:
    """Linear-interpolate frames where all landmarks are zero (MediaPipe dropout).
    arr: (T, n_landmarks, axes)
    """
    norms = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
    missing = norms < zero_threshold
    if not missing.any():
        return arr
    out = arr.copy()
    valid_idx = np.where(~missing)[0]
    if len(valid_idx) == 0:
        return out
    for t in np.where(missing)[0]:
        before = valid_idx[valid_idx < t]
        after  = valid_idx[valid_idx > t]
        if len(before) == 0:
            out[t] = out[after[0]]
        elif len(after) == 0:
            out[t] = out[before[-1]]
        else:
            t0, t1 = before[-1], after[0]
            alpha = (t - t0) / (t1 - t0)
            out[t] = (1 - alpha) * arr[t0] + alpha * arr[t1]
    return out


def _interpolate_missing_pose(pose: np.ndarray, zero_threshold: float = 1e-4) -> np.ndarray:
    """Pose landmark 개별 보간: landmark별로 (0,0,0)인 프레임을 보간.

    학습 데이터(AI Hub)는 pose 랜드마크가 거의 항상 검출되지만,
    브라우저 MediaPipe는 손목 등이 미검출 시 (0,0,0)을 반환.
    정규화 전 보간하여 학습 데이터 분포와 일치시킴.

    pose: (T, 9, 3)
    """
    T = pose.shape[0]
    out = pose.copy()
    for lm_idx in range(pose.shape[1]):
        col = pose[:, lm_idx, :]                          # (T, 3)
        norms = np.linalg.norm(col, axis=-1)              # (T,)
        missing = norms < zero_threshold
        if not missing.any():
            continue
        valid_idx = np.where(~missing)[0]
        if len(valid_idx) == 0:
            continue
        for t in np.where(missing)[0]:
            before = valid_idx[valid_idx < t]
            after  = valid_idx[valid_idx > t]
            if len(before) == 0:
                out[t, lm_idx] = out[after[0], lm_idx]
            elif len(after) == 0:
                out[t, lm_idx] = out[before[-1], lm_idx]
            else:
                t0, t1 = before[-1], after[0]
                alpha = (t - t0) / (t1 - t0)
                out[t, lm_idx] = (1 - alpha) * pose[t0, lm_idx] + alpha * pose[t1, lm_idx]
    return out


def convert_stream(keypoints: np.ndarray, target_preset: str = "B") -> np.ndarray:
    """Convert frontend keypoints (N, 210) → (N, feature_dim) without temporal interpolation.

    학습 preprocess.py의 load_npz_keypoints와 동일한 파이프라인:
      1. 손 누락 프레임 보간 (all-zero → interpolate)
      2. pose 랜드마크 개별 보간 (브라우저 MediaPipe 미검출 → 특이값 방지)
      3. 어깨 기준 정규화
      4. preset B: xy축 + pose 7개(hip 제외)
    """
    cfg = FEATURE_PRESETS[target_preset]
    n_frames = keypoints.shape[0]

    full_landmarks = keypoints.reshape(n_frames, 70, 3)

    offsets_raw = {}
    running = 0
    for name in LANDMARK_ORDER:
        offsets_raw[name] = running
        running += _FRONTEND_COUNTS[name]

    # 손 누락 프레임 보간 (frame 전체가 all-zero인 경우)
    for name in ("left_hand", "right_hand"):
        s = offsets_raw[name]
        e = s + _FRONTEND_COUNTS[name]
        full_landmarks[:, s:e, :] = _interpolate_missing(full_landmarks[:, s:e, :])

    # pose 랜드마크 개별 보간 (landmark별 미검출 → 특이값 방지)
    ps = offsets_raw["pose"]
    pe = ps + _FRONTEND_COUNTS["pose"]
    full_landmarks[:, ps:pe, :] = _interpolate_missing_pose(full_landmarks[:, ps:pe, :])

    full_landmarks = _normalize_by_shoulders(full_landmarks)

    parts = []
    for name in cfg["use"]:
        start = offsets_raw[name]
        count = LANDMARK_COUNTS[name]
        group = full_landmarks[:, start:start + count, :cfg["axes"]]
        parts.append(group.reshape(n_frames, -1))

    return np.concatenate(parts, axis=-1).astype(np.float32)


def convert_keypoints(
    keypoints: np.ndarray,
    target_preset: str,
    target_seq_len: int,
) -> np.ndarray:
    """Convert frontend keypoints (N, 210) to model input (target_seq_len, feature_dim).

    Frontend always sends 70 landmarks x 3 axes = 210 dim in order:
    [pose(9x3), left_hand(21x3), right_hand(21x3), face(19x3)]
    """
    cfg = FEATURE_PRESETS[target_preset]
    n_frames = keypoints.shape[0]

    full_landmarks = keypoints.reshape(n_frames, 70, 3)

    # hand keypoint 누락 프레임 보간
    offsets = {}
    running = 0
    for name in LANDMARK_ORDER:
        offsets[name] = running
        running += _FRONTEND_COUNTS[name]
    for name in ("left_hand", "right_hand"):
        s = offsets[name]
        e = s + _FRONTEND_COUNTS[name]
        full_landmarks[:, s:e, :] = _interpolate_missing(full_landmarks[:, s:e, :])

    ps = offsets["pose"]
    pe = ps + _FRONTEND_COUNTS["pose"]
    full_landmarks[:, ps:pe, :] = _interpolate_missing_pose(full_landmarks[:, ps:pe, :])

    full_landmarks = _normalize_by_shoulders(full_landmarks)

    parts = []
    for name in cfg["use"]:
        start = offsets[name]
        count = LANDMARK_COUNTS[name]   # 실제 사용할 개수 (pose=7, 나머지 전체)
        group = full_landmarks[:, start:start + count, :cfg["axes"]]
        parts.append(group.reshape(n_frames, -1))

    converted = np.concatenate(parts, axis=-1).astype(np.float32)
    return temporal_interpolate(converted, target_seq_len)
