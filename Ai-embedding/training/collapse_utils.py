"""
collapse_utils.py — dimensional collapse 진단/방지 유틸.

Phase 1 진단에서 데모 인코더가 256-d 중 effective rank ~23 으로 collapse 됨을 확인.
loss 는 0.03 까지 떨어지는데 LOO Top-3 는 75% 에서 정체 = collapse 시그니처.

여기 제공:
  - effective_rank(Z)          : 학습 중 collapse 를 *보이게* 하는 모니터링 지표
  - vicreg_var_cov(z)          : variance hinge + covariance 패널티 (Bardes et al. ICLR'22)
  - uniformity_loss(z)         : Wang & Isola (ICML'20) — 대안/보조

설계 노트:
  - VICReg 항은 *정규화前* 피처(combined)에 건다. 단위구로 사영된 뒤엔 per-dim
    variance 가 구속돼 variance hinge 가 무의미해지기 때문.
  - uniformity 항은 *정규화後* 단위벡터에 건다(초구 위 분포 균일화).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def effective_rank(Z: torch.Tensor):
    """임베딩 행렬 (N, D) 의 collapse 정도를 두 지표로 반환.

    participation_ratio = (Σλ)² / Σλ²   (고유값 λ = 특이값²)
    entropy_rank        = exp(H(p)),  p = λ/Σλ   (Roy & Vetterli effective rank)
    둘 다 D 에 가까울수록 건강, 1 에 가까울수록 완전 collapse.
    """
    Zc = Z - Z.mean(dim=0, keepdim=True)
    # 특이값 (float32 CPU 가 MPS svd 보다 안정적)
    s = torch.linalg.svdvals(Zc.float().cpu())
    lam = s ** 2
    tot = lam.sum().clamp_min(1e-12)
    pr = float((tot ** 2) / (lam ** 2).sum().clamp_min(1e-12))
    p = lam / tot
    ent = -(p * (p + 1e-12).log()).sum()
    return {"participation_ratio": pr,
            "entropy_rank": float(ent.exp()),
            "top1_var_frac": float(p.max()),
            "dim": int(Z.shape[1])}


def vicreg_var_cov(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4):
    """VICReg 의 variance + covariance 항 (invariance 항은 NT-Xent 가 담당).

    z: (N, D) — *정규화前* 임베딩.
      variance: 각 차원 표준편차가 gamma 미만이면 hinge 패널티 → 죽은 차원 부활.
      covariance: 차원 간 공분산 off-diagonal 제곱합 → 차원 중복(collapse) 억제.
    반환: (var_loss, cov_loss)
    """
    N, D = z.shape
    # variance hinge
    std = torch.sqrt(z.var(dim=0) + eps)            # (D,)
    var_loss = F.relu(gamma - std).mean()
    # covariance off-diagonal
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(N - 1, 1)               # (D, D)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = off.pow(2).sum() / D
    return var_loss, cov_loss


def uniformity_loss(z: torch.Tensor, t: float = 2.0):
    """Wang & Isola (ICML'20): 단위구 위 균일성. z 는 *정규화後* (N, D) 단위벡터.
    log E[exp(-t * ||zi - zj||²)] — 작을수록 더 균일하게 퍼짐."""
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return sq_pdist.mul(-t).exp().mean().clamp_min(1e-12).log()
