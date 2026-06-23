import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import SignEncoder, VelocityGatedSignEncoder, LandmarkStreamEncoder, LandmarkCrossStreamEncoder


def _load_state(path: str):
    """Load checkpoint, handling both raw state_dict and wrapped checkpoint formats."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "encoder_state_dict" in state:
        return state["encoder_state_dict"]
    return state


def _load_meta(path: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v for k, v in state.items() if k != "encoder_state_dict"}


class EmbeddingService:
    def __init__(self, model_path: str, input_dim: int):
        self.device = torch.device("cpu")
        self.encoder = SignEncoder(input_dim=input_dim)
        self.encoder.load_state_dict(_load_state(model_path))
        self.encoder.eval()
        self.encoder.to(self.device)
        self.input_dim = input_dim

    @staticmethod
    def infer_input_dim(model_path: str) -> int:
        state = _load_state(model_path)
        return state["input_proj.weight"].shape[1]

    @torch.no_grad()
    def embed(self, keypoints: np.ndarray) -> np.ndarray:
        """Embed preprocessed keypoints (seq_len, input_dim) -> (d_model,) float32."""
        tensor = torch.from_numpy(keypoints).unsqueeze(0).to(self.device)
        emb = self.encoder(tensor)
        return emb.cpu().numpy().astype(np.float32).squeeze(0)


class EmbeddingServiceExp026:
    """EXP-026 인코더 서비스 (norm_first=True, d_model=256, fixed 64-frame)."""

    TARGET_LENGTH = 64  # DB 빌드와 동일한 고정 길이

    def __init__(self, model_path: str):
        self.device = torch.device("cpu")
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        from preprocess import feature_dim_for
        state = ckpt.get("encoder_state_dict", ckpt)
        mtype      = ckpt.get("model_type", "")
        is_cross   = mtype == "landmark_cross_stream" or "cross_attn.norm1.weight" in state
        is_stream  = mtype == "landmark_stream" or ("stream_logits" in state and not is_cross)
        is_velgate = any("vel_gate" in k for k in state)

        if is_cross or is_stream:
            d_model    = ckpt.get("d_model", 256)
            nl = sum(1 for k in state if k.startswith("encoders.0.") and "self_attn.in_proj_weight" in k)
            num_layers = nl if nl > 0 else ckpt.get("num_layers", 2)
            if is_cross:
                self.encoder = LandmarkCrossStreamEncoder(
                    d_model=d_model, nhead=8, num_layers=num_layers,
                    dim_feedforward=512, dropout=0.0, norm_first=True, cross_nhead=4,
                )
            else:
                self.encoder = LandmarkStreamEncoder(
                    d_model=d_model, nhead=8, num_layers=num_layers,
                    dim_feedforward=512, dropout=0.0, norm_first=True,
                )
            self.input_dim = 136
            self.d_model   = d_model
            self._is_stream = True
        else:
            input_dim  = ckpt.get("input_dim",  state["input_proj.weight"].shape[1])
            d_model    = ckpt.get("d_model",    state["input_proj.weight"].shape[0])
            num_layers = ckpt.get("num_layers", sum(1 for k in state if "self_attn.in_proj_weight" in k))
            EncoderCls = VelocityGatedSignEncoder if is_velgate else SignEncoder
            self.encoder = EncoderCls(
                input_dim=input_dim, d_model=d_model, nhead=8,
                num_layers=num_layers, dim_feedforward=1024,
                dropout=0.0, norm_first=True,
            )
            self.input_dim = input_dim
            self.d_model   = d_model
            self._is_stream = False

        self.encoder.load_state_dict(state)
        self.encoder.eval()

    @torch.no_grad()
    def embed_varlen(self, segment: np.ndarray) -> np.ndarray:
        """Embed a segment (T, input_dim) -> (d_model,) float32.
        Always interpolates to TARGET_LENGTH=64 to match DB build conditions.
        """
        from preprocess import temporal_interpolate
        seg = temporal_interpolate(segment.astype(np.float32), self.TARGET_LENGTH)
        x = torch.from_numpy(seg).unsqueeze(0)  # (1, 64, input_dim)
        if self._is_stream:
            emb = self.encoder(x)
        else:
            emb = self.encoder(x, src_key_padding_mask=None)
        return F.normalize(emb[0], dim=0).cpu().numpy().astype(np.float32)
