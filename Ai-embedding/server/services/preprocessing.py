"""
Landmark-keyed keypoints → model feature vector.

Node.js sends each word segment as a dict with landmark keys:
  {
    "pose":       (T, 27) — 9 landmarks × 3 axes
    "left_hand":  (T, 63) — 21 × 3
    "right_hand": (T, 63) — 21 × 3
    "face":       (T, 57) — 19 × 3
  }

Output: (T, feature_dim) — preset B = 136 (7+21+21+19 landmarks × 2 axes)
"""
import numpy as np

# Counts sent by Node.js per landmark group (per frame, flattened)
_INPUT_DIMS = {
    "pose":       9  * 3,   # 27
    "left_hand":  21 * 3,   # 63
    "right_hand": 21 * 3,   # 63
    "face":       19 * 3,   # 57
}
_INPUT_LM_COUNTS = {"pose": 9, "left_hand": 21, "right_hand": 21, "face": 19}

# Counts used by the model (pose: drop 2 hip landmarks → 7 remain)
_MODEL_LM_COUNTS = {"pose": 7, "left_hand": 21, "right_hand": 21, "face": 19}

FEATURE_PRESETS = {
    "A": {"use": ["pose", "left_hand", "right_hand", "face"], "axes": 3},
    "B": {"use": ["pose", "left_hand", "right_hand", "face"], "axes": 2},
    "C": {"use": ["pose", "left_hand", "right_hand"],         "axes": 2},
}

# Shoulder landmark indices within the pose block (L_sh=1, R_sh=2)
_SHOULDER_L, _SHOULDER_R = 1, 2


def feature_dim_for(preset: str) -> int:
    cfg = FEATURE_PRESETS[preset]
    return sum(_MODEL_LM_COUNTS[k] for k in cfg["use"]) * cfg["axes"]


# ── Missing frame interpolation ───────────────────────────────────────────────

def _interpolate_missing(arr: np.ndarray, zero_threshold: float = 1e-4) -> np.ndarray:
    """Linear-interpolate frames where all landmarks are zero (MediaPipe dropout).
    arr: (T, n_landmarks, 3)
    """
    norms   = np.linalg.norm(arr.reshape(arr.shape[0], -1), axis=1)
    missing = norms < zero_threshold
    if not missing.any():
        return arr
    out       = arr.copy()
    valid_idx = np.where(~missing)[0]
    if len(valid_idx) == 0:
        return out
    for t in np.where(missing)[0]:
        before = valid_idx[valid_idx < t]
        after  = valid_idx[valid_idx > t]
        if   len(before) == 0: out[t] = out[after[0]]
        elif len(after)  == 0: out[t] = out[before[-1]]
        else:
            t0, t1 = before[-1], after[0]
            alpha  = (t - t0) / (t1 - t0)
            out[t] = (1 - alpha) * arr[t0] + alpha * arr[t1]
    return out


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize(pose: np.ndarray, parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Center on shoulder midpoint and scale by shoulder width.

    pose: (T, 9, 3)  — used only to read shoulder positions
    parts: all landmark groups as (T, n_lm, 3) arrays
    Returns normalized copies of parts.
    """
    l_sh  = pose[:, _SHOULDER_L, :2]   # (T, 2)
    r_sh  = pose[:, _SHOULDER_R, :2]
    mid   = (l_sh + r_sh) / 2.0        # (T, 2)
    width = np.maximum(np.linalg.norm(r_sh - l_sh, axis=-1, keepdims=True), 1e-6)  # (T, 1)

    out = {}
    for name, arr in parts.items():
        a       = arr.copy()
        a[..., 0] = (arr[..., 0] - mid[:, 0:1]) / width
        a[..., 1] = (arr[..., 1] - mid[:, 1:2]) / width
        a[..., 2] =  arr[..., 2]                / width
        out[name] = a
    return out


# ── Main converter ────────────────────────────────────────────────────────────

def convert_word(word: dict[str, list], preset: str = "B") -> np.ndarray:
    """Convert one landmark-keyed word segment to a model feature array.

    Args:
        word: {
            "pose":       (T, 27) flat list-of-lists  [9 lm × 3 axes]
            "left_hand":  (T, 63)                     [21 × 3]
            "right_hand": (T, 63)                     [21 × 3]
            "face":       (T, 57)                     [19 × 3]
        }
        preset: "A" | "B" | "C"

    Returns:
        np.ndarray of shape (T, feature_dim), dtype float32
    """
    cfg = FEATURE_PRESETS[preset]

    # Parse each group into (T, n_lm, 3) arrays
    parts: dict[str, np.ndarray] = {}
    T = None
    for name in ("pose", "left_hand", "right_hand", "face"):
        raw = np.array(word[name], dtype=np.float32)          # (T, n*3)
        n   = _INPUT_LM_COUNTS[name]
        arr = raw.reshape(raw.shape[0], n, 3)                 # (T, n, 3)
        if T is None:
            T = arr.shape[0]
        parts[name] = arr

    # Interpolate missing hand frames (MediaPipe dropout → all-zero)
    for name in ("left_hand", "right_hand"):
        parts[name] = _interpolate_missing(parts[name])

    # Normalize: shoulder-center + shoulder-width scale
    parts = _normalize(parts["pose"], parts)

    # Select landmarks and axes per preset
    feature_parts = []
    for name in cfg["use"]:
        n_keep = _MODEL_LM_COUNTS[name]             # drop trailing lm if needed (pose: 9→7)
        group  = parts[name][:, :n_keep, :cfg["axes"]]   # (T, n_keep, axes)
        feature_parts.append(group.reshape(T, -1))

    return np.concatenate(feature_parts, axis=-1).astype(np.float32)


def temporal_interpolate(seq: np.ndarray, target: int) -> np.ndarray:
    T = seq.shape[0]
    if T == target:
        return seq.astype(np.float32)
    if T == 0:
        raise ValueError("empty sequence")
    if T == 1:
        return np.repeat(seq, target, axis=0).astype(np.float32)
    idx = np.linspace(0, T - 1, target)
    return np.array([seq[int(round(i))] for i in idx], dtype=np.float32)
