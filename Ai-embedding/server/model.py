"""SignEncoder + ProjectionHead + MomentumEncoder + BYOLPredictor."""
import copy
import math

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
    """단일 랜드마크 그룹 전용 인코더 (LandmarkStreamEncoder / LandmarkCrossStreamEncoder 서브모듈)."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = self.transformer(x)
        return x[:, 0, :]


class LandmarkStreamEncoder(nn.Module):
    """4-stream landmark encoder with learnable weighted average (face 가중치 낮게 초기화)."""

    _STREAM_DIMS  = [14, 42, 42, 38]
    _SLICES       = [(0, 14), (14, 56), (56, 98), (98, 136)]
    _INIT_LOGITS  = [1.0, 1.5, 1.5, 0.3]

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 2,
                 dim_feedforward: int = 512, dropout: float = 0.1, norm_first: bool = True):
        super().__init__()
        self.encoders = nn.ModuleList([
            StreamEncoder(dim, d_model, nhead, num_layers, dim_feedforward, dropout, norm_first)
            for dim in self._STREAM_DIMS
        ])
        self.stream_logits = nn.Parameter(torch.tensor(self._INIT_LOGITS, dtype=torch.float32))
        self.d_model = d_model

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        w = F.normalize(self.stream_logits.softmax(dim=0), dim=0)
        combined = None
        for i, (s, e) in enumerate(self._SLICES):
            emb = self.encoders[i](x[:, :, s:e])
            combined = w[i] * emb if combined is None else combined + w[i] * emb
        return F.normalize(combined, dim=-1)


class CrossStreamAttention(nn.Module):
    """스트림 임베딩 간 self-attention (B, N_streams, D) → (B, N_streams, D)."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1     = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2     = nn.LayerNorm(d_model)
        self.ff        = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class LandmarkCrossStreamEncoder(nn.Module):
    """4-stream + Cross-stream Attention 인코더 (최종 채택 모델)."""

    _STREAM_DIMS  = [14, 42, 42, 38]
    _SLICES       = [(0, 14), (14, 56), (56, 98), (98, 136)]
    _INIT_LOGITS  = [1.0, 1.5, 1.5, 0.3]

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 2,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 norm_first: bool = True, cross_nhead: int = 4):
        super().__init__()
        self.encoders = nn.ModuleList([
            StreamEncoder(dim, d_model, nhead, num_layers, dim_feedforward, dropout, norm_first)
            for dim in self._STREAM_DIMS
        ])
        self.cross_attn    = CrossStreamAttention(d_model, nhead=cross_nhead, dropout=dropout)
        self.stream_logits = nn.Parameter(torch.tensor(self._INIT_LOGITS, dtype=torch.float32))
        self.d_model = d_model

    def forward(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        stream_embs = [self.encoders[i](x[:, :, s:e]) for i, (s, e) in enumerate(self._SLICES)]
        stacked  = torch.stack(stream_embs, dim=1)
        fused    = self.cross_attn(stacked)
        w        = self.stream_logits.softmax(dim=0)
        combined = (fused * w.view(1, 4, 1)).sum(dim=1)
        return F.normalize(combined, dim=-1)


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
