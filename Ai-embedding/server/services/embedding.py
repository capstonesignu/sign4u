"""
Sign encoder — wraps SignEncoder (project root model.py).

Loads any checkpoint produced by the AI-Embedding training pipeline:
  - raw state_dict (result_varlen_baseline, result_fixed_baseline, …)
  - wrapped {"encoder_state_dict": ..., "input_dim": ..., ...}

Architecture parameters (num_layers, d_model, input_dim) are inferred
from the state dict so any compatible checkpoint can be used without
changing config.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SERVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_ROOT))

from model import SignEncoder, VelocityGatedSignEncoder, LandmarkStreamEncoder, LandmarkCrossStreamEncoder
from services.preprocessing import temporal_interpolate

TARGET_LENGTH = 64   # must match the DB build target length


class Encoder:
    def __init__(self, model_path: str):
        ckpt  = torch.load(model_path, map_location="cpu", weights_only=False)
        state = ckpt.get("encoder_state_dict", ckpt)
        mtype = ckpt.get("model_type", "")

        is_cross   = mtype == "landmark_cross_stream" or "cross_attn.norm1.weight" in state
        is_stream  = mtype == "landmark_stream" or ("stream_logits" in state and not is_cross)
        is_velgate = any("vel_gate" in k for k in state)

        if is_cross or is_stream:
            d_model    = ckpt.get("d_model", 256)
            nl = sum(1 for k in state if k.startswith("encoders.0.") and "self_attn.in_proj_weight" in k)
            num_layers = nl if nl > 0 else ckpt.get("num_layers", 2)
            if is_cross:
                self.model = LandmarkCrossStreamEncoder(
                    d_model=d_model, nhead=8, num_layers=num_layers,
                    dim_feedforward=512, dropout=0.0, norm_first=True, cross_nhead=4,
                )
            else:
                self.model = LandmarkStreamEncoder(
                    d_model=d_model, nhead=8, num_layers=num_layers,
                    dim_feedforward=512, dropout=0.0, norm_first=True,
                )
            self.input_dim  = 136
            self.d_model    = d_model
            self.num_layers = num_layers
            self._is_stream = True
        else:
            input_dim  = ckpt.get("input_dim",  state["input_proj.weight"].shape[1])
            d_model    = ckpt.get("d_model",    state["input_proj.weight"].shape[0])
            num_layers = ckpt.get("num_layers", sum(1 for k in state if "self_attn.in_proj_weight" in k))
            EncoderCls = VelocityGatedSignEncoder if is_velgate else SignEncoder
            self.model = EncoderCls(
                input_dim=input_dim, d_model=d_model, nhead=8,
                num_layers=num_layers, dim_feedforward=1024,
                dropout=0.0, norm_first=True,
            )
            self.input_dim  = input_dim
            self.d_model    = d_model
            self.num_layers = num_layers
            self._is_stream = False

        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def embed(self, segment: np.ndarray) -> np.ndarray:
        """(T, input_dim) → (d_model,) L2-normalized float32."""
        seg = temporal_interpolate(segment.astype(np.float32), TARGET_LENGTH)
        x   = torch.from_numpy(seg).unsqueeze(0)
        emb = self.model(x) if self._is_stream else self.model(x, src_key_padding_mask=None)
        return F.normalize(emb[0], dim=0).cpu().numpy().astype(np.float32)
