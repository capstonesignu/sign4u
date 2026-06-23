"""
compare_encoders.py — 기존 vs collapse-fix 인코더 before/after 비교.

Phase 3 의 진짜 목표는 LOO 숫자가 아니라 false-confidence 해소다.
동일 데이터(recorded-video, 평가 전용)를 두 인코더로 임베딩해
collapse / same-diff 분리 / cross-word 충돌 / DB-LOO margin 을 직접 비교한다.

데이터 규칙: recorded-video 는 평가 전용 — 여기서도 *평가용 임베딩*으로만 쓴다(학습 없음).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python compare_encoders.py
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

import eval_tta_loo as E                       # load_recorded, embed_with_tta, encoder cls
from collapse_utils import effective_rank
from diagnose_embedding import auc_mannwhitney, best_threshold, pct

ENCODERS = {
    "기존(prod)": "encoder/encoder_cross_stream_best.pt",
    "신규(collapse-fix ep50)": "result_cs_collapse/encoder_cross_stream_best.pt",
}


def embed_all(model_path, recorded, device):
    enc = E.LandmarkCrossStreamEncoder(d_model=256, nhead=8, num_layers=2,
                                       dim_feedforward=512, dropout=0.0, norm_first=True).to(device)
    ck = torch.load(model_path, map_location="cpu", weights_only=False)
    enc.load_state_dict(ck.get("encoder_state_dict", ck), strict=True)
    enc.eval()
    vecs, labels = [], []
    for w, segs in recorded.items():
        for s in segs:
            vecs.append(E.embed_with_tta(enc, s, device, n_tta=1))
            labels.append(w)
    V = np.stack(vecs).astype(np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return V, np.array(labels)


def analyze(V, labels, coll_thr=0.9):
    n = len(labels)
    sim = V @ V.T
    triu = np.triu(np.ones((n, n), bool), 1)
    same = (labels[:, None] == labels[None, :])
    same_s = sim[same & triu]; diff_s = sim[(~same) & triu]
    sep_auc = auc_mannwhitney(same_s, diff_s)
    rk = effective_rank(torch.from_numpy(V))
    n_coll = int(((~same) & triu & (sim > coll_thr)).sum())

    # DB-LOO score/margin
    S = sim.copy(); np.fill_diagonal(S, -np.inf)
    cor_s, inc_s, cor_m, inc_m = [], [], [], []
    correct = 0
    for i in range(n):
        best = {}
        for j in range(n):
            if j == i: continue
            w = labels[j]; v = float(S[i, j])
            if w not in best or v > best[w]: best[w] = v
        ranked = sorted(best.values(), reverse=True)
        ok = (max(best, key=best.get) == labels[i]); correct += ok
        s1 = ranked[0]; m12 = (s1 - ranked[1]) if len(ranked) > 1 else 0.0
        (cor_s if ok else inc_s).append(s1)
        (cor_m if ok else inc_m).append(m12)
    cor_s, inc_s = np.array(cor_s), np.array(inc_s)
    cor_m, inc_m = np.array(cor_m), np.array(inc_m)
    return {
        "PR": rk["participation_ratio"], "entropy_rank": rk["entropy_rank"],
        "top1_var": rk["top1_var_frac"],
        "diff_mean": float(diff_s.mean()), "diff_p99": pct(diff_s, 99),
        "sep_auc": sep_auc, "n_collision": n_coll,
        "loo_top1": correct / n,
        "score_auc": auc_mannwhitney(cor_s, inc_s),
        "margin_auc": auc_mannwhitney(cor_m, inc_m),
        "score_gap": (cor_s.mean() - inc_s.mean()) if len(cor_s) and len(inc_s) else float("nan"),
        "margin_thr": best_threshold(cor_m, inc_m)[0],
    }


def main():
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    recorded = E.load_recorded(E.RECORDED_DIR, target_length=E.TARGET_LENGTH)
    print(f"[data] recorded-video {len(recorded)}단어 "
          f"{sum(len(v) for v in recorded.values())}샘플 · device={device}\n")

    results = {}
    for name, path in ENCODERS.items():
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} 없음"); continue
        V, labels = embed_all(path, recorded, device)
        results[name] = analyze(V, labels)

    rows = [
        ("effective rank PR /256",  "PR",          "{:.1f}", "↑"),
        ("entropy rank",            "entropy_rank","{:.1f}", "↑"),
        ("top1 특이값 분산",         "top1_var",    "{:.1%}", "↓"),
        ("diff-word 평균 cos",       "diff_mean",   "{:.3f}", "↓"),
        ("diff-word p99 cos",        "diff_p99",    "{:.3f}", "↓"),
        ("same>diff 분리 AUC",       "sep_auc",     "{:.3f}", "↑"),
        ("cross-word 충돌(cos>0.9)", "n_collision", "{:d}",   "↓"),
        ("LOO top1 정확도",          "loo_top1",    "{:.1%}", "↑"),
        ("절대 score reject AUC",    "score_auc",   "{:.3f}", "↑"),
        ("margin reject AUC",        "margin_auc",  "{:.3f}", "↑"),
        ("margin 임계 θ",            "margin_thr",  "{:.3f}", "·"),
    ]
    names = list(results.keys())
    w0 = 26
    print(f"{'지표':<{w0}}" + "".join(f"{n:>26}" for n in names) + "   (방향)")
    print("─" * (w0 + 26 * len(names) + 9))
    for label, key, fmt, arrow in rows:
        cells = ""
        for n in names:
            v = results[n][key]
            cells += f"{(fmt.format(int(v)) if fmt=='{:d}' else fmt.format(v)):>26}"
        print(f"{label:<{w0}}{cells}   {arrow}")


if __name__ == "__main__":
    main()
