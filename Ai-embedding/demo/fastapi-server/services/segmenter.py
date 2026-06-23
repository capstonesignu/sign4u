import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from seg_model import BiLSTMSegmenter


class SegmenterService:
    def __init__(self, model_path: str):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = BiLSTMSegmenter(
            input_dim=ckpt.get("input_dim", 136),
            hidden=ckpt.get("hidden", 256),
            n_layers=ckpt.get("n_layers", 3),
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        print(
            f"[SegmenterService] loaded epoch={ckpt.get('epoch','?')} "
            f"F1={ckpt.get('best_val_f1', '?')}"
        )

    def predict_segments(
        self,
        stream: np.ndarray,
        b_threshold: float = 0.80,
        min_segment_frames: int = 6,
    ) -> list[tuple[int, int]]:
        """(T, 136) numpy array → [(start, end), ...] inclusive pairs."""
        x = torch.tensor(stream, dtype=torch.float32)
        return self.model.predict_segments(
            x,
            b_threshold=b_threshold,
            min_segment_frames=min_segment_frames,
        )
