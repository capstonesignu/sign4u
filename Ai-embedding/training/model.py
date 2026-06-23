"""SignEncoder + ProjectionHead + MomentumEncoder + BYOLPredictor."""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIM = 150
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FF = 1024
DROPOUT = 0.1
PROJ_DIM = 64
MAX_LEN = 512


class PositionalEncoding(nn.Module):
    """
    역할: Transformer 입력에 위치 정보 추가
    수식: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int = D_MODEL, max_len: int = MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력: (batch, seq, d_model) → 위치 정보 추가."""
        return x + self.pe[:, : x.size(1)]


class SignEncoder(nn.Module):
    """
    역할: 키포인트 시퀀스 → 임베딩 벡터
    구조: Linear + PosEnc + CLS + TransformerEncoder → L2 정규화
    가변 길이 지원: src_key_padding_mask로 패딩 구간 마스킹
    """

    def __init__(self,
                 input_dim: int = INPUT_DIM,
                 d_model: int = D_MODEL,
                 nhead: int = NHEAD,
                 num_layers: int = NUM_LAYERS,
                 dim_feedforward: int = DIM_FF,
                 dropout: float = DROPOUT,
                 norm_first: bool = False):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        입력: (batch, T, input_dim)  — T는 고정(64) 또는 가변
        mask: (batch, T) bool — True인 위치는 패딩 (attention 제외)
        출력: (batch, d_model) L2 정규화
        """
        B = x.size(0)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if src_key_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        else:
            full_mask = None
        x = self.transformer(x, src_key_padding_mask=full_mask)
        return F.normalize(x[:, 0, :], dim=-1)


class VelocityGate(nn.Module):
    """Preset B 전용 landmark-group velocity gating.

    각 프레임의 landmark group별 움직임 크기를 계산해 learned gate를 통해
    input_proj 출력을 조절한다.

    - 고속 (실제 수화 동작): gate > 0 → embedding 증폭
    - 저속 (멈춤 / 보간 0 / 카메라 밖): gate < 0 → embedding 억제
    - zero-init으로 시작 → 훈련 초기엔 원래 SignEncoder와 동일 동작

    Groups (preset B, 136-dim):
        pose       [0:14]   7 lm × 2
        left_hand  [14:56] 21 lm × 2
        right_hand [56:98] 21 lm × 2
        face       [98:136] 19 lm × 2
    """

    _GROUPS = [(0, 14), (14, 56), (56, 98), (98, 136)]
    N_GROUPS = 4

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(self.N_GROUPS, d_model)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, x: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, T, d_model) — input_proj 이후 token embeddings
        seq: (B, T, D)       — 원본 keypoint sequence (D=136)
        반환: (B, T, d_model) — gated embeddings with residual
        """
        vels = []
        for s, e in self._GROUPS:
            diff = seq[:, 1:, s:e] - seq[:, :-1, s:e]   # (B, T-1, group_d)
            vel  = torch.norm(diff, dim=-1)               # (B, T-1)
            vel  = torch.cat([vel[:, :1], vel], dim=1)    # (B, T)  첫 프레임 복사
            vels.append(vel)

        vel_feat = torch.stack(vels, dim=-1)              # (B, T, 4)

        # sequence-level 정규화: max 기준 [0, 1]
        v_max    = vel_feat.amax(dim=1, keepdim=True).clamp(min=1e-6)
        vel_feat = vel_feat / v_max

        gate = torch.tanh(self.gate_proj(vel_feat))       # (B, T, d_model)
        return x + x * gate                               # residual: x·(1 + gate)


class VelocityGatedSignEncoder(nn.Module):
    """VelocityGate를 포함한 SignEncoder.

    SignEncoder와 동일한 인터페이스를 유지하되,
    input_proj 직후 VelocityGate를 삽입한다.
    """

    def __init__(self,
                 input_dim: int = INPUT_DIM,
                 d_model: int = D_MODEL,
                 nhead: int = NHEAD,
                 num_layers: int = NUM_LAYERS,
                 dim_feedforward: int = DIM_FF,
                 dropout: float = DROPOUT,
                 norm_first: bool = False):
        super().__init__()
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.vel_gate     = VelocityGate(d_model)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        입력: (batch, T, input_dim)
        mask: (batch, T) bool — True = padding
        출력: (batch, d_model) L2 정규화
        """
        B   = x.size(0)
        raw = x                                              # 원본 보존 (gate 계산용)
        x   = self.input_proj(x)
        x   = self.vel_gate(x, raw)                         # velocity gating
        x   = self.pos_encoding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        if src_key_padding_mask is not None:
            cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        else:
            full_mask = None
        x = self.transformer(x, src_key_padding_mask=full_mask)
        return F.normalize(x[:, 0, :], dim=-1)


class StreamEncoder(nn.Module):
    """단일 랜드마크 그룹 전용 인코더 (LandmarkStreamEncoder 서브모듈)."""

    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int,
                 dropout: float, norm_first: bool):
        super().__init__()
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """(B, T, input_dim) → (B, d_model)  — 정규화 전 raw CLS 출력."""
        B = x.size(0)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        if src_key_padding_mask is not None:
            cls_mask  = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)
        else:
            full_mask = None
        x = self.transformer(x, src_key_padding_mask=full_mask)
        return x[:, 0, :]


class LandmarkStreamEncoder(nn.Module):
    """4-stream landmark encoder with learnable weighted average.

    Preset B (136-dim) 분리:
      pose        [0:14]   7 lm × 2
      left_hand  [14:56]  21 lm × 2
      right_hand [56:98]  21 lm × 2
      face       [98:136] 19 lm × 2

    각 스트림이 독립 Transformer로 특화 학습.
    stream_logits → softmax → 가중 평균 → L2 정규화.
    face 초기 가중치를 낮게 설정해 얼굴 랜드마크 과의존 방지.
    """

    _STREAM_DIMS  = [14, 42, 42, 38]          # preset B 각 그룹 차원
    _STREAM_NAMES = ["pose", "left_hand", "right_hand", "face"]
    _SLICES = [(0, 14), (14, 56), (56, 98), (98, 136)]

    # face 초기 가중치: pose=1.0, left=1.5, right=1.5, face=0.3
    _INIT_LOGITS = [1.0, 1.5, 1.5, 0.3]

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 norm_first: bool = True):
        super().__init__()
        self.encoders = nn.ModuleList([
            StreamEncoder(
                input_dim=dim, d_model=d_model, nhead=nhead,
                num_layers=num_layers, dim_feedforward=dim_feedforward,
                dropout=dropout, norm_first=norm_first,
            )
            for dim in self._STREAM_DIMS
        ])
        self.stream_logits = nn.Parameter(
            torch.tensor(self._INIT_LOGITS, dtype=torch.float32)
        )
        self.d_model = d_model

    def stream_weights(self) -> torch.Tensor:
        """현재 softmax 가중치 반환 (디버깅용)."""
        return F.softmax(self.stream_logits, dim=0)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        입력: (B, T, 136)  — Preset B 전체 feature
        출력: (B, d_model) L2 정규화
        """
        w = F.softmax(self.stream_logits, dim=0)   # (4,) 합=1
        combined = None
        for i, (s, e) in enumerate(self._SLICES):
            stream_emb = self.encoders[i](x[:, :, s:e])  # (B, d_model)
            weighted   = w[i] * stream_emb
            combined   = weighted if combined is None else combined + weighted
        return F.normalize(combined, dim=-1)


class CrossStreamAttention(nn.Module):
    """스트림 임베딩 간 self-attention (B, N_streams, D) → (B, N_streams, D).

    각 스트림 임베딩을 하나의 토큰으로 취급해 스트림 간 상호작용을 학습.
    예: 손 스트림이 얼굴 스트림을 참조 → 귀/눈/코 혼동 감소 기대.

    Pre-LN Transformer block (1-layer): Self-Attn → Add&Norm → FFN → Add&Norm
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2     = nn.LayerNorm(d_model)
        self.ff        = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N_streams, D) → (B, N_streams, D)"""
        # Pre-LN Self-Attention + residual
        h, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        # Pre-LN FFN + residual
        x = x + self.ff(self.norm2(x))
        return x


class LandmarkCrossStreamEncoder(nn.Module):
    """LandmarkStreamEncoder + Cross-stream Attention.

    각 스트림이 독립 Transformer로 인코딩 후,
    스트림 임베딩 간 self-attention으로 cross-stream 상호작용 학습.
      - hand 스트림 ↔ face 스트림: 손의 얼굴 근접 위치 파악
      - left ↔ right: 양손 협응 동작 파악
    최종: weighted average → L2 정규화.

    Preset B (136-dim):
      pose        [0:14]   7 lm × 2
      left_hand  [14:56]  21 lm × 2
      right_hand [56:98]  21 lm × 2
      face       [98:136] 19 lm × 2
    """

    _STREAM_DIMS  = [14, 42, 42, 38]
    _STREAM_NAMES = ["pose", "left_hand", "right_hand", "face"]
    _SLICES       = [(0, 14), (14, 56), (56, 98), (98, 136)]
    _INIT_LOGITS  = [1.0, 1.5, 1.5, 0.3]   # face 낮게 초기화

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 2,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 norm_first: bool = True,
                 cross_nhead: int = 4):
        super().__init__()
        self.encoders = nn.ModuleList([
            StreamEncoder(
                input_dim=dim, d_model=d_model, nhead=nhead,
                num_layers=num_layers, dim_feedforward=dim_feedforward,
                dropout=dropout, norm_first=norm_first,
            )
            for dim in self._STREAM_DIMS
        ])
        self.cross_attn    = CrossStreamAttention(d_model, nhead=cross_nhead, dropout=dropout)
        self.stream_logits = nn.Parameter(
            torch.tensor(self._INIT_LOGITS, dtype=torch.float32)
        )
        self.d_model = d_model

    def stream_weights(self) -> torch.Tensor:
        return F.softmax(self.stream_logits, dim=0)

    def forward(self, x: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None,
                return_prenorm: bool = False):
        """
        입력: (B, T, 136)
        출력: (B, d_model) L2 정규화
        return_prenorm=True 면 (정규화 임베딩, 정규화前 combined) 튜플 반환.
          VICReg variance/covariance 항은 단위구로 사영되기 *전* 피처에 걸어야 하므로
          학습 시에만 사용. 추론/검색 경로는 기존과 동일(단일 정규화 텐서).
        """
        # 1) 스트림별 독립 인코딩 (padding mask 전달)
        stream_embs = []
        for i, (s, e) in enumerate(self._SLICES):
            emb = self.encoders[i](x[:, :, s:e],
                                   src_key_padding_mask=src_key_padding_mask)
            stream_embs.append(emb)

        # 2) (B, 4, D) 로 stack → cross-stream attention
        stacked = torch.stack(stream_embs, dim=1)   # (B, 4, D)
        fused   = self.cross_attn(stacked)           # (B, 4, D)

        # 3) 가중 평균 → L2 정규화
        w       = F.softmax(self.stream_logits, dim=0)   # (4,)
        combined = (fused * w.view(1, 4, 1)).sum(dim=1)  # (B, D)
        normed   = F.normalize(combined, dim=-1)
        if return_prenorm:
            return normed, combined
        return normed


class VelocityLandmarkCrossStreamEncoder(LandmarkCrossStreamEncoder):
    """LandmarkCrossStreamEncoder + 명시적 velocity 피처.

    입력: (B, T, 272) — 각 프레임 [pos(136) | vel(136)]
    vel[t] = pos[t] - pos[t-1]  (t=0은 0으로 패딩)

    스트림별 슬라이스 (pos+vel 연속 배치):
      pose      [0:28]   pos[0:14]  + vel[0:14]
      left_hand [28:112] pos[14:56] + vel[14:56]
      right_hand[112:196]pos[56:98] + vel[56:98]
      face      [196:272]pos[98:136]+ vel[98:136]
    """
    _STREAM_DIMS  = [28, 84, 84, 76]
    _STREAM_NAMES = ["pose", "left_hand", "right_hand", "face"]
    _SLICES       = [(0, 28), (28, 112), (112, 196), (196, 272)]
    _INIT_LOGITS  = [1.0, 1.5, 1.5, 0.3]


def add_velocity(seq: np.ndarray) -> np.ndarray:
    """(T, 136) → (T, 272): 각 스트림 pos 뒤에 vel 연접.

    출력 레이아웃:
      [pose_pos(0:14) | pose_vel(14:28) | lhand_pos(28:70) | lhand_vel(70:112)
       | rhand_pos(112:154) | rhand_vel(154:196) | face_pos(196:234) | face_vel(234:272)]
    """
    T, D = seq.shape
    assert D == 136, f"add_velocity expects 136-dim input, got {D}"
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]

    slices = [(0, 14), (14, 56), (56, 98), (98, 136)]
    parts = []
    for s, e in slices:
        parts.append(seq[:, s:e])
        parts.append(vel[:, s:e])
    return np.concatenate(parts, axis=-1).astype(np.float32)


class ProjectionHead(nn.Module):
    """
    역할: Exp 4 (D-Q-X-M) 전용 Projection Head
    구조: Linear → ReLU → Linear → L2 정규화
    """

    def __init__(self, d_model: int = D_MODEL, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력: (batch, 128) → 출력: (batch, 64) L2 정규화."""
        return F.normalize(self.fc(x), dim=-1)


class MomentumEncoder(nn.Module):
    """
    역할: Online encoder의 EMA target (MoCo / BYOL 공용)
    업데이트: θ_ξ ← m · θ_ξ + (1 - m) · θ_θ
    특징: gradient 없음, forward 시 no_grad
    """

    def __init__(self, online_module: nn.Module, m: float = 0.999):
        super().__init__()
        self.module = copy.deepcopy(online_module)
        for p in self.module.parameters():
            p.requires_grad = False
        self.m = m

    @torch.no_grad()
    def update(self, online_module: nn.Module) -> None:
        """EMA 업데이트."""
        for p_t, p_o in zip(self.module.parameters(),
                            online_module.parameters()):
            p_t.data.mul_(self.m).add_(p_o.data, alpha=1.0 - self.m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.module(x)


class BYOLPredictor(nn.Module):
    """
    역할: BYOL online branch predictor h_θ
    구조: Linear → BatchNorm → ReLU → Linear → L2 정규화
    """

    def __init__(self, dim: int = PROJ_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


if __name__ == "__main__":
    # CPU-only forward-pass sanity test
    torch.manual_seed(0)

    batch_size = 2
    enc = SignEncoder()
    proj = ProjectionHead()
    enc.eval()
    proj.eval()

    x = torch.randn(batch_size, 64, INPUT_DIM)
    with torch.no_grad():
        z = enc(x)
        p = proj(z)

    print(f"[model] input={tuple(x.shape)} enc={tuple(z.shape)} proj={tuple(p.shape)}")
    print(f"[model] enc L2 norm={z.norm(dim=-1).tolist()}")
    print(f"[model] proj L2 norm={p.norm(dim=-1).tolist()}")

    assert z.shape == (batch_size, D_MODEL), "SignEncoder output shape mismatch"
    assert p.shape == (batch_size, PROJ_DIM), "ProjectionHead output shape mismatch"
    assert torch.allclose(z.norm(dim=-1), torch.ones(batch_size), atol=1e-5), \
        "SignEncoder output not L2-normalized"
    assert torch.allclose(p.norm(dim=-1), torch.ones(batch_size), atol=1e-5), \
        "ProjectionHead output not L2-normalized"

    n_enc = sum(pp.numel() for pp in enc.parameters())
    n_proj = sum(pp.numel() for pp in proj.parameters())
    print(f"[model] params enc={n_enc:,} proj={n_proj:,}")
    print("[model] sanity test passed.")
