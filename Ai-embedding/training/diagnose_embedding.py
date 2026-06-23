"""
diagnose_embedding.py — Phase 1 embedding-space 진단.

기사(uniformity/collapse, margin-based confidence)에서 지적한 "다른 동작인데
confidence 높음(false confidence)" 증상의 근본 원인을 *고치기 전에 측정*한다.

대상: 데모(인식 테스트 탭)가 실제로 서빙하는 프로덕션 쌍
  - 인코더: encoder/encoder_cross_stream_best.pt  (LandmarkCrossStreamEncoder, NT-Xent)
  - DB:     db/jbedu_recordings.faiss             (IndexFlatIP, 256-d, L2-normalized)

다섯 가지 진단을 한 번에 수행하고 diagnostics/ 아래에 그림 + 텍스트 리포트를 남긴다:
  1a. same-class vs different-class 코사인 분포 분리도   → uniformity / 임베딩 문제 여부
  1b. 특이값 스펙트럼 + effective rank                  → dimensional collapse (Jing et al. ICLR'22)
  *.  cross-word near-duplicate 감사                    → flip/혼동 벡터가 false-match를 만드는지
  1c. recorded-video 쿼리의 top1 score / margin 분포     → 인식 정직성, Phase 2 임계값 ROC 보정
  요약. 정상/위험 판정 + 권고

데이터 규칙: dataset/recorded-video/ 는 *평가 전용* — 절대 학습/DB에 투입하지 않는다.
            본 스크립트는 그것을 *쿼리*로만 사용한다(학습 없음).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python diagnose_embedding.py
    python diagnose_embedding.py --encoder encoder/encoder_cross_stream_best.pt \
        --faiss db/jbedu_recordings.faiss --out diagnostics
    python diagnose_embedding.py --no-query        # DB 구조만(인코더/쿼리 스킵, 빠름)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import faiss
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# eval_tta_loo 가 sys.path 에 demo/fastapi-server 를 넣고 model/preprocess 를 import 한다.
# 프로덕션과 동일 경로를 재사용하기 위해 그 모듈을 통째로 빌려 쓴다.
import eval_tta_loo as E


# ──────────────────────────────────────────────────────────────────────────────
# 작은 유틸 (sklearn 의존 없이 AUC / threshold)
# ──────────────────────────────────────────────────────────────────────────────
def auc_mannwhitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """pos(정답) 점수가 neg(오답)보다 높을 확률 = ROC-AUC. tie=0.5."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # tie 평균 랭크
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def best_threshold(pos: np.ndarray, neg: np.ndarray):
    """Youden's J 를 최대화하는 임계값(>=thr → accept) 과 그때의 TPR/FPR."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan"), float("nan")
    cands = np.unique(np.concatenate([pos, neg]))
    best_j, best_t, best_tpr, best_fpr = -1.0, cands[0], 0.0, 0.0
    for t in cands:
        tpr = float((pos >= t).mean())
        fpr = float((neg >= t).mean())
        j = tpr - fpr
        if j > best_j:
            best_j, best_t, best_tpr, best_fpr = j, float(t), tpr, fpr
    return best_t, best_tpr, best_fpr


def pct(a: np.ndarray, q) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ──────────────────────────────────────────────────────────────────────────────
# 진단
# ──────────────────────────────────────────────────────────────────────────────
def load_db(faiss_path: str, labels_path: str):
    index = faiss.read_index(faiss_path)
    n, d = index.ntotal, index.d
    vecs = np.zeros((n, d), dtype="float32")
    for i in range(n):
        vecs[i] = index.reconstruct(i)
    with open(labels_path, "r") as f:
        labels = json.load(f)
    assert len(labels) == n, f"labels({len(labels)}) != ntotal({n})"
    # 안전: 재정규화(이미 정규화돼 있어야 함)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return index, vecs, np.array(labels)


def diag_same_diff(vecs, labels, out: Path, rpt):
    sim = vecs @ vecs.T
    n = len(labels)
    eye = np.eye(n, dtype=bool)
    same = (labels[:, None] == labels[None, :]) & ~eye
    diff = (labels[:, None] != labels[None, :])
    # 상삼각만 (중복 쌍 제거)
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    same_sims = sim[same & triu]
    diff_sims = sim[diff & triu]

    d = vecs.shape[1]
    rand_mean, rand_std = 0.0, 1.0 / np.sqrt(d)  # 랜덤 단위벡터 기준선

    auc = auc_mannwhitney(same_sims, diff_sims)  # same 가 diff 보다 큰 정도(분리도)
    overlap = float((diff_sims > np.median(same_sims)).mean()) if len(same_sims) else float("nan")

    rpt.append("── 1a. same-class vs different-class 코사인 분포 ──────────────────")
    rpt.append(f"  벡터 {n}개 · 차원 {d} · 단어(고유) {len(set(labels.tolist()))}개")
    rpt.append(f"  same-class 쌍 {len(same_sims)}개 :  mean={same_sims.mean():.4f}  "
               f"std={same_sims.std():.4f}  p5={pct(same_sims,5):.4f}  p50={pct(same_sims,50):.4f}")
    rpt.append(f"  diff-class 쌍 {len(diff_sims)}개 :  mean={diff_sims.mean():.4f}  "
               f"std={diff_sims.std():.4f}  p95={pct(diff_sims,95):.4f}  p99={pct(diff_sims,99):.4f}")
    rpt.append(f"  랜덤 단위벡터 기준선(d={d}): mean≈{rand_mean:.3f}  std≈{rand_std:.4f}")
    rpt.append(f"  분리도 AUC(same>diff)        : {auc:.4f}   (1.0=완전분리, 0.5=구분불가)")
    rpt.append(f"  겹침: median(same)보다 큰 diff쌍 비율 = {overlap:.1%}")
    verdict = ("정상" if diff_sims.mean() < 0.3 and auc > 0.9
               else "주의" if diff_sims.mean() < 0.5
               else "위험(uniformity 낮음/collapse 의심)")
    rpt.append(f"  판정: diff 평균 {diff_sims.mean():.3f} → {verdict}")
    rpt.append("")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.2, 1.0, 80)
    ax.hist(diff_sims, bins=bins, alpha=0.6, density=True, label=f"different-word (μ={diff_sims.mean():.3f})", color="#ef4444")
    ax.hist(same_sims, bins=bins, alpha=0.6, density=True, label=f"same-word (μ={same_sims.mean():.3f})", color="#22c55e")
    ax.axvline(rand_mean, ls="--", c="#888", lw=1, label="random baseline (≈0)")
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("density")
    ax.set_title("1a. same vs different-word similarity (DB internal)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "1a_same_vs_diff.png", dpi=120); plt.close(fig)
    return sim, diff & triu


def diag_collapse(vecs, out: Path, rpt):
    d = vecs.shape[1]
    centered = vecs - vecs.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    lam = s ** 2
    pr = float((lam.sum() ** 2) / (lam ** 2).sum())          # participation ratio(고유값)
    p = lam / lam.sum()
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))  # effective rank(엔트로피)
    cum = np.cumsum(lam) / lam.sum()
    dims90 = int(np.searchsorted(cum, 0.90) + 1)
    dims99 = int(np.searchsorted(cum, 0.99) + 1)

    rpt.append("── 1b. dimensional collapse (특이값 스펙트럼) ────────────────────")
    rpt.append(f"  전체 차원 d={d}")
    rpt.append(f"  participation ratio(고유값) : {pr:.1f} / {d}")
    rpt.append(f"  effective rank(엔트로피)     : {eff_rank:.1f} / {d}")
    rpt.append(f"  분산 90% 설명에 필요한 차원   : {dims90}")
    rpt.append(f"  분산 99% 설명에 필요한 차원   : {dims99}")
    rpt.append(f"  top-1 특이값이 차지하는 분산   : {p[0]:.1%}")
    verdict = ("정상" if pr > d * 0.4 else "주의" if pr > d * 0.15 else "위험(dimensional collapse)")
    rpt.append(f"  판정: PR {pr:.0f}/{d} → {verdict}")
    rpt.append("")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(s, marker=".", ms=3); a1.set_yscale("log")
    a1.set_xlabel("dim index"); a1.set_ylabel("singular value (log)")
    a1.set_title(f"1b. singular spectrum  (PR={pr:.0f}/{d})")
    a2.plot(cum, lw=1.5); a2.axhline(0.9, ls="--", c="#888", lw=1)
    a2.set_xlabel("dim index"); a2.set_ylabel("cumulative variance")
    a2.set_title("cumulative explained variance")
    fig.tight_layout(); fig.savefig(out / "1b_collapse.png", dpi=120); plt.close(fig)


def diag_nearcollision(sim, diff_mask, labels, out: Path, rpt, thr=0.9, topn=25):
    """다른 단어인데 코사인>thr 인 near-duplicate 쌍 — false-match 의 직접 원인."""
    ii, jj = np.where(diff_mask & (sim > thr))
    rpt.append("── *. cross-word near-duplicate 감사 (다른 단어, cos>%.2f) ──────────" % thr)
    rpt.append(f"  해당 쌍 수: {len(ii)}  (전체 diff 쌍 대비 {len(ii)}/{int(diff_mask.sum())})")
    if len(ii):
        vals = sim[ii, jj]
        order = np.argsort(-vals)[:topn]
        rpt.append(f"  상위 {min(topn,len(order))}개 충돌 (서로 다른 단어인데 거의 동일 임베딩):")
        for k in order:
            a, b = labels[ii[k]], labels[jj[k]]
            rpt.append(f"    cos={vals[k]:.3f}   {a}  ~  {b}")
        # 단어쌍별 집계
        from collections import Counter
        pairc = Counter(tuple(sorted((labels[ii[k]], labels[jj[k]]))) for k in range(len(ii)))
        rpt.append(f"  혼동 단어쌍 종류: {len(pairc)}  (상위 5: "
                   + ", ".join(f"{a}~{b}×{c}" for (a, b), c in pairc.most_common(5)) + ")")
        verdict = "위험(혼동 무리 존재 → reject 게이트로도 못 거름)" if len(pairc) > 5 else "주의"
    else:
        verdict = "정상(다른 단어 간 near-duplicate 없음)"
    rpt.append(f"  판정: {verdict}")
    rpt.append("")


def diag_db_loo(vecs, labels, sim, out: Path, rpt):
    """DB 내부 LOO: 각 벡터를 빼고 나머지서 검색 → 데모 DB 자체의 score/margin ROC.
    recorded-video 와 어휘가 분리돼 있어, Phase 2 임계값 보정의 *올바른* 입력은 이것."""
    n = len(labels)
    S = sim.copy()
    np.fill_diagonal(S, -np.inf)
    cor_s, inc_s, cor_m12, inc_m12, cor_m1m, inc_m1m = ([] for _ in range(6))
    correct_n = 0
    for i in range(n):
        row = S[i]
        best = {}
        for j in range(n):
            if j == i:
                continue
            w = labels[j]; s = float(row[j])
            if w not in best or s > best[w]:
                best[w] = s
        ranked = sorted(best.values(), reverse=True)
        if not ranked:
            continue
        top1w = max(best, key=best.get)
        ok = (top1w == labels[i])
        correct_n += ok
        s1 = ranked[0]
        m12 = (s1 - ranked[1]) if len(ranked) > 1 else None
        tail = ranked[1:5]
        m1m = (s1 - float(np.mean(tail))) if tail else None
        (cor_s if ok else inc_s).append(s1)
        if m12 is not None:
            (cor_m12 if ok else inc_m12).append(m12)
        if m1m is not None:
            (cor_m1m if ok else inc_m1m).append(m1m)
    cor_s, inc_s = np.array(cor_s), np.array(inc_s)
    cor_m12, inc_m12 = np.array(cor_m12), np.array(inc_m12)

    rpt.append("── 1c'. DB 내부 LOO — score/margin ROC (데모 DB 자체) ────────────")
    rpt.append(f"  LOO top1 정확도: {correct_n}/{n} = {correct_n/n:.1%}")
    if len(cor_s) and len(inc_s):
        auc_s = auc_mannwhitney(cor_s, inc_s)
        t_s, tpr_s, fpr_s = best_threshold(cor_s, inc_s)
        rpt.append(f"  top1 score : 정답 mean={cor_s.mean():.3f}  오답 mean={inc_s.mean():.3f}  "
                   f"AUC={auc_s:.3f}")
        rpt.append(f"             → θ_abs≈{t_s:.3f}  (정답 통과 TPR={tpr_s:.0%}, 오답 통과 FPR={fpr_s:.0%})")
    if len(cor_m12) and len(inc_m12):
        auc_m = auc_mannwhitney(cor_m12, inc_m12)
        t_m, tpr_m, fpr_m = best_threshold(cor_m12, inc_m12)
        rpt.append(f"  margin12   : 정답 mean={cor_m12.mean():.3f}  오답 mean={inc_m12.mean():.3f}  "
                   f"AUC={auc_m:.3f}")
        rpt.append(f"             → θ_mar≈{t_m:.3f}  (TPR={tpr_m:.0%}, FPR={fpr_m:.0%})")
        better = "margin" if auc_m > auc_s else "score"
        rpt.append(f"  → 더 잘 거르는 신호: {better}.  이 θ 가 Phase 2 reject 게이트 시작값.")
    rpt.append("")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    if len(cor_s) or len(inc_s):
        b = np.linspace(0, 1, 40)
        a1.hist(inc_s, bins=b, alpha=0.6, density=True, color="#ef4444", label=f"오답 n={len(inc_s)}")
        a1.hist(cor_s, bins=b, alpha=0.6, density=True, color="#22c55e", label=f"정답 n={len(cor_s)}")
        a1.set_title("1c'. DB-LOO top1 score"); a1.set_xlabel("cosine"); a1.legend(fontsize=8)
    if len(cor_m12) or len(inc_m12):
        b = np.linspace(0, 0.6, 40)
        a2.hist(inc_m12, bins=b, alpha=0.6, density=True, color="#ef4444", label="오답")
        a2.hist(cor_m12, bins=b, alpha=0.6, density=True, color="#22c55e", label="정답")
        a2.set_title("1c'. DB-LOO margin (top1-top2)"); a2.set_xlabel("margin"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "1c_db_loo.png", dpi=120); plt.close(fig)


def diag_sameword_audit(vecs, labels, sim, out: Path, rpt, low_thr=0.3):
    """같은 단어 벡터쌍의 intra 코사인 분포. 낮은 쌍이 많으면 DB에 flip/노이즈 혼입 의심.
    flip 시그니처: 한 단어 내부가 bimodal(절반 높고 절반 낮음)."""
    from collections import defaultdict
    by_word = defaultdict(list)
    for i, w in enumerate(labels):
        by_word[w].append(i)

    intra_mins, intra_all, low_words = [], [], []
    bimodal = []
    for w, idxs in by_word.items():
        if len(idxs) < 2:
            continue
        pair_sims = []
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                pair_sims.append(float(sim[idxs[a], idxs[b]]))
        pair_sims = np.array(pair_sims)
        intra_all.extend(pair_sims.tolist())
        intra_mins.append(pair_sims.min())
        if pair_sims.min() < low_thr:
            low_words.append((w, len(idxs), float(pair_sims.min()), float(pair_sims.max())))
        # bimodal: 높은(>0.7) 쌍과 낮은(<0.4) 쌍이 둘 다 존재
        if (pair_sims > 0.7).any() and (pair_sims < 0.4).any():
            bimodal.append(w)

    intra_all = np.array(intra_all)
    rpt.append("── 2. same-word intra-class 코사인 감사 (flip/노이즈 혼입) ──────────")
    rpt.append(f"  다중녹화 단어 {len(intra_mins)}개 · intra 쌍 {len(intra_all)}개")
    rpt.append(f"  intra 코사인: mean={intra_all.mean():.3f}  p5={pct(intra_all,5):.3f}  "
               f"p50={pct(intra_all,50):.3f}")
    rpt.append(f"  intra-min < {low_thr} 인 단어: {len(low_words)}개")
    for w, k, mn, mx in sorted(low_words, key=lambda x: x[2])[:15]:
        rpt.append(f"    {w} (n={k}): min={mn:.3f}  max={mx:.3f}")
    rpt.append(f"  bimodal(>0.7 & <0.4 공존) 단어 {len(bimodal)}개"
               + (": " + ", ".join(bimodal[:12]) if bimodal else ""))
    n_neg = sum(1 for m in intra_mins if m < 0.0)  # 같은 단어인데 음수 코사인 = 강한 적신호
    if n_neg > 0 or len(bimodal) > len(intra_mins) * 0.1:
        v = (f"위험(같은 단어인데 음수 코사인 {n_neg}개 · bimodal {len(bimodal)}개 "
             "→ flip 벡터 DB 혼입 또는 손상 녹화 강력 의심)")
    elif len(low_words) > len(intra_mins) * 0.2:
        v = "주의(노이즈 큰 녹화 다수 — 같은 단어 일관성 낮음)"
    else:
        v = "정상"
    rpt.append(f"  판정: {v}")
    rpt.append("")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(intra_all, bins=np.linspace(-0.2, 1.0, 60), color="#3b82f6", alpha=0.8)
    ax.axvline(low_thr, ls="--", c="#ef4444", lw=1, label=f"low_thr={low_thr}")
    ax.set_title("2. same-word intra-class cosine"); ax.set_xlabel("cosine"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "2_sameword_intra.png", dpi=120); plt.close(fig)


def diag_query_margin(args, out: Path, rpt):
    import torch
    device = (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    encoder = E.LandmarkCrossStreamEncoder(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_ff, dropout=0.0, norm_first=True,
    ).to(device)
    ckpt = torch.load(args.encoder, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt.get("encoder_state_dict", ckpt), strict=True)
    encoder.eval()

    index = faiss.read_index(args.faiss)
    with open(args.labels, "r") as f:
        labels = json.load(f)
    db_words = set(labels)

    recorded = E.load_recorded(args.recorded_dir, target_length=E.TARGET_LENGTH)
    n_samp = sum(len(v) for v in recorded.values())
    in_db = [w for w in recorded if w in db_words]

    rows = []  # (word, correct(bool), in_db(bool), top1, m12, m1mean)
    for word, segs in recorded.items():
        for seg in segs:
            q = E.embed_with_tta(encoder, seg, device, n_tta=1).reshape(1, -1).copy()
            faiss.normalize_L2(q)
            D, I = index.search(q, min(50, index.ntotal))
            best = {}
            for s, idx in zip(D[0], I[0]):
                if 0 <= idx < len(labels):
                    w = labels[idx]
                    if w not in best or s > best[w]:
                        best[w] = float(s)
            ranked = sorted(best.values(), reverse=True)
            top1w = max(best, key=best.get)
            s1 = ranked[0]
            s2 = ranked[1] if len(ranked) > 1 else None
            m12 = (s1 - s2) if s2 is not None else None
            tail = ranked[1:5]
            m1mean = (s1 - float(np.mean(tail))) if tail else None
            rows.append((word, top1w == word, word in db_words, s1, m12, m1mean))

    arr_indb = [r for r in rows if r[2]]  # ROC 는 DB에 단어가 있는 쿼리에 한해서만 의미
    cor = np.array([r[3] for r in arr_indb if r[1]])
    inc = np.array([r[3] for r in arr_indb if not r[1]])
    m12_cor = np.array([r[4] for r in arr_indb if r[1] and r[4] is not None])
    m12_inc = np.array([r[4] for r in arr_indb if not r[1] and r[4] is not None])

    acc = float(np.mean([r[1] for r in arr_indb])) if arr_indb else float("nan")
    rpt.append("── 1c. recorded-video 쿼리 → 프로덕션 DB 검색 (margin/score) ──────")
    rpt.append(f"  recorded: {len(recorded)}단어 {n_samp}샘플 · 그중 DB에 존재하는 단어 {len(in_db)}개")
    rpt.append(f"  in-DB 쿼리 {len(arr_indb)}개 LOO-style top1 정확도: {acc:.1%}")
    rpt.append(f"  top1 score   정답 mean={cor.mean():.3f}  오답 mean={inc.mean():.3f}" if len(cor) and len(inc) else "  (정답/오답 표본 부족)")
    if len(cor) and len(inc):
        auc_s = auc_mannwhitney(cor, inc)
        t_s, tpr_s, fpr_s = best_threshold(cor, inc)
        rpt.append(f"  top1 score   분리 AUC={auc_s:.3f}  → reject 임계 θ_abs≈{t_s:.3f} "
                   f"(TPR={tpr_s:.0%}, FPR={fpr_s:.0%})")
    if len(m12_cor) and len(m12_inc):
        auc_m = auc_mannwhitney(m12_cor, m12_inc)
        t_m, tpr_m, fpr_m = best_threshold(m12_cor, m12_inc)
        rpt.append(f"  margin12     정답 mean={m12_cor.mean():.3f}  오답 mean={m12_inc.mean():.3f}  "
                   f"AUC={auc_m:.3f}  → θ_mar≈{t_m:.3f} (TPR={tpr_m:.0%}, FPR={fpr_m:.0%})")
        rpt.append("  → 위 θ_abs/θ_mar 가 Phase 2 reject 게이트의 ROC-보정 시작값.")
    rpt.append("  주의(기대치): margin reject 는 인식률을 올리는 게 아니라 모호한 입력을 "
               "정직히 거부 → recall↓ precision↑. 진짜 해결은 학습(Phase 3).")
    rpt.append("")

    # 그림: score / margin, 정답 vs 오답
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    if len(cor) or len(inc):
        b = np.linspace(0, 1, 40)
        a1.hist(inc, bins=b, alpha=0.6, density=True, color="#ef4444", label=f"오답 n={len(inc)}")
        a1.hist(cor, bins=b, alpha=0.6, density=True, color="#22c55e", label=f"정답 n={len(cor)}")
        a1.set_title("1c. top1 cosine score"); a1.set_xlabel("score"); a1.legend(fontsize=8)
    if len(m12_cor) or len(m12_inc):
        b = np.linspace(0, 0.6, 40)
        a2.hist(m12_inc, bins=b, alpha=0.6, density=True, color="#ef4444", label="오답")
        a2.hist(m12_cor, bins=b, alpha=0.6, density=True, color="#22c55e", label="정답")
        a2.set_title("1c. margin (top1-top2)"); a2.set_xlabel("margin"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "1c_query_margin.png", dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="encoder/encoder_cross_stream_best.pt")
    p.add_argument("--faiss",   default="db/jbedu_recordings.faiss")
    p.add_argument("--labels",  default="db/jbedu_recordings.faiss.labels.json")
    p.add_argument("--recorded-dir", default="dataset/recorded-video")
    p.add_argument("--out",     default="diagnostics")
    p.add_argument("--collision-thr", type=float, default=0.9)
    p.add_argument("--no-query", action="store_true", help="DB 구조만 진단(인코더/쿼리 스킵)")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--nhead",   type=int, default=8)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dim-ff",  type=int, default=512)
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rpt = ["================ Phase 1 Embedding 진단 리포트 ================",
           f"encoder: {args.encoder}", f"faiss:   {args.faiss}", ""]

    index, vecs, labels = load_db(args.faiss, args.labels)
    sim, diff_mask = diag_same_diff(vecs, labels, out, rpt)
    diag_collapse(vecs, out, rpt)
    diag_nearcollision(sim, diff_mask, labels, out, rpt, thr=args.collision_thr)
    diag_db_loo(vecs, labels, sim, out, rpt)
    diag_sameword_audit(vecs, labels, sim, out, rpt)

    if not args.no_query:
        try:
            diag_query_margin(args, out, rpt)
        except Exception as e:
            rpt.append(f"── 1c. 쿼리 진단 실패: {type(e).__name__}: {e}")
            rpt.append("")

    text = "\n".join(rpt)
    print(text)
    (out / "phase1_report.txt").write_text(text, encoding="utf-8")
    print(f"\n[saved] {out}/phase1_report.txt  +  1a/1b/1c PNG")


if __name__ == "__main__":
    main()
