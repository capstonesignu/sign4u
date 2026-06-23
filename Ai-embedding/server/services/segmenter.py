"""
Word boundary segmenter — two modes:

  BiLSTM mode : loaded from SEGMENTER_PATH when the file exists.
  Velocity mode: pure-numpy pause detection; no model file needed.

Both return list[(start_frame, end_frame)] inclusive.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Velocity-based (always available) ────────────────────────────────────────

def velocity_segment(
    stream: np.ndarray,
    threshold: float = 0.05,
    pause_frames: int = 8,
    min_seg_frames: int = 8,
) -> list[tuple[int, int]]:
    """Pause-based segmenter on pose+hand dims (98-dim prefix of stream).

    Args:
        stream: (T, feature_dim) preprocessed keypoint stream.
    Returns:
        List of (start, end) inclusive frame index pairs.
    """
    T = stream.shape[0]
    if T < 2:
        return [(0, T - 1)]

    dims       = stream[:, :98]                              # pose + hands only
    vel        = np.linalg.norm(np.diff(dims, axis=0), axis=1)  # (T-1,)
    vel_smooth = np.convolve(vel, np.ones(3) / 3, mode="same")
    is_pause   = vel_smooth < threshold

    boundaries: list[int] = []
    pause_start: Optional[int] = None
    for i, p in enumerate(is_pause):
        if p:
            if pause_start is None:
                pause_start = i
        else:
            if pause_start is not None:
                length = i - pause_start
                if length >= pause_frames:
                    boundaries.append(pause_start + length // 2)
                pause_start = None
    if pause_start is not None:
        length = len(is_pause) - pause_start
        if length >= pause_frames:
            boundaries.append(pause_start + length // 2)

    if not boundaries:
        return [(0, T - 1)]

    cuts  = [0] + [b + 1 for b in boundaries] + [T]
    spans = [
        (cuts[i], cuts[i + 1] - 1)
        for i in range(len(cuts) - 1)
        if cuts[i + 1] - 1 - cuts[i] + 1 >= min_seg_frames
    ]
    return spans or [(0, T - 1)]


# ── BiLSTM-based (optional) ───────────────────────────────────────────────────

class BiLSTMSegmenter:
    def __init__(self, model_path: str):
        import torch
        from seg_model import BiLSTMSegmenter as _Model  # noqa: F401

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self._model = _Model(
            input_dim=ckpt.get("input_dim", 136),
            hidden=ckpt.get("hidden", 256),
            n_layers=ckpt.get("n_layers", 3),
        )
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()
        self._torch = torch
        print(
            f"[BiLSTMSegmenter] epoch={ckpt.get('epoch','?')} "
            f"F1={ckpt.get('best_val_f1','?')}"
        )

    def predict(
        self,
        stream: np.ndarray,
        b_threshold: float = 0.80,
        min_segment_frames: int = 6,
    ) -> list[tuple[int, int]]:
        """(T, 136) → [(start, end), ...]."""
        x = self._torch.tensor(stream, dtype=self._torch.float32)
        return self._model.predict_segments(
            x,
            b_threshold=b_threshold,
            min_segment_frames=min_segment_frames,
        )


def load_segmenter(model_path: str) -> Optional[BiLSTMSegmenter]:
    """Returns BiLSTMSegmenter if model file exists, else None (use velocity_segment)."""
    if not Path(model_path).exists():
        print(f"[segmenter] Model not found at {model_path} — using velocity segmenter")
        return None
    try:
        return BiLSTMSegmenter(model_path)
    except Exception as e:
        print(f"[segmenter] Load failed ({e}) — using velocity segmenter")
        return None
