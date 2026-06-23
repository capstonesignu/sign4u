"""
train_cs_collapse.py — CrossStream + HNM + collapse 방지(VICReg) + effective-rank 로깅.

Phase 1 진단 결과 대응:
  - 데모 인코더가 256-d 중 effective rank ~23 으로 dimensional collapse.
  - NT-Xent loss 는 0.03 까지 떨어지는데 LOO Top-3 는 75% 정체 (collapse 시그니처).
  - 기존 train_cs_hnm.py 에는 collapse 를 *보는* 지표조차 없었음.

이 스크립트의 변경(= train_cs_hnm 대비):
  1. VICReg variance+covariance 항을 *정규화前* 임베딩에 추가 → 죽은 차원 부활/중복 억제.
  2. (선택) uniformity 항을 정규화後 임베딩에 추가.
  3. 매 val 마다 effective_rank(PR / entropy_rank) 로깅 → collapse 가 보임.
  4. best 선택은 기존대로 recorded LOO Top-3.
  데이터/증강/HNM/eval 은 train_cs_hnm 의 함수를 그대로 import 해 사용(중복 방지).

데이터 규칙: dataset/recorded-video/ 는 평가 전용. 여기서도 LOO 평가에만 쓰고 학습 금지.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python train_cs_collapse.py
    python train_cs_collapse.py --epochs 2 --steps 10 --val-every 1   # smoke test
    python train_cs_collapse.py --lambda-var 1.0 --lambda-cov 0.04    # 항 가중치 스윕
"""
from __future__ import annotations
import argparse, json, os, random, time
from typing import Dict, List, Set

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F

import train_cs_hnm as H   # 모든 데이터/증강/HNM/eval 헬퍼 재사용 (sys.path/import 부작용 포함)
from train_cs_hnm import (
    LandmarkCrossStreamEncoder, TARGET_LENGTH,
    WORD_MAP_PATH, DIR_WORDS_PATH, AIHUB_DIR, RECORDED_DIR, N_HOLDOUT, SEED, CKPT_NAME,
)
from collapse_utils import effective_rank, vicreg_var_cov, uniformity_loss

OUT_DIR = "result_cs_collapse"


@torch.no_grad()
def measure_rank(encoder, segs: Dict[str, List[np.ndarray]], device, max_samples=600):
    """val/train segs 일부를 임베딩(정규화後) → effective_rank. Phase 1 진단과 동일 기준."""
    encoder.eval()
    embs, count = [], 0
    for w, ss in segs.items():
        for s in ss:
            x = torch.from_numpy(s).unsqueeze(0).to(device)
            embs.append(F.normalize(encoder(x)[0], dim=0).cpu())
            count += 1
            if count >= max_samples:
                break
        if count >= max_samples:
            break
    Z = torch.stack(embs)
    return effective_rank(Z)


def train(args):
    os.makedirs(args.output, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    word_map = json.load(open(WORD_MAP_PATH, encoding="utf-8"))
    directional: Set[str] = set()
    if os.path.exists(DIR_WORDS_PATH):
        directional = set(json.load(open(DIR_WORDS_PATH, encoding="utf-8")))

    print("[0] recorded-video 로딩 (평가 전용)...")
    recorded = H.load_recorded(args.recorded_dir, target_length=args.target_length,
                               fps_norm=args.fps_norm, target_fps=args.target_fps)
    print(f"     {len(recorded)} words, {sum(len(v) for v in recorded.values())} samples")

    print("[1] AI Hub 로딩...")
    all_segs = H.load_all_segs(word_map, args.aihub_dir, target_length=args.target_length)
    holdout_words, train_words = H.split_holdout(sorted(all_segs.keys()), N_HOLDOUT)

    rng_inst = random.Random(SEED + 1)
    train_segs: Dict[str, List[np.ndarray]] = {}
    val_segs:   Dict[str, List[np.ndarray]] = {}
    for w in train_words:
        if w not in all_segs:
            continue
        insts = all_segs[w][:]
        rng_inst.shuffle(insts)
        n_val = max(1, len(insts) // 5)
        val_segs[w]   = insts[:n_val]
        train_segs[w] = insts[n_val:] if len(insts[n_val:]) >= 2 else insts
    print(f"[split] train={sum(len(v) for v in train_segs.values())} segs")

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[device] {device}")

    encoder = LandmarkCrossStreamEncoder(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        dropout=args.dropout, norm_first=True,
    ).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[model] d_model={args.d_model} L={args.num_layers} params={n_params:,}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * args.steps)

    best_rec_top3 = -1.0
    patience_count = 0
    history = []
    train_wl = list(train_segs.keys())
    val_wl   = list(val_segs.keys())
    hard_negs: Dict[str, List[str]] = {}
    warp_range = (args.warp_min, args.warp_max)

    print(f"\n[train] {args.epochs}ep × {args.steps}steps  lr={args.lr}  temp={args.temperature}  "
          f"λvar={args.lambda_var} λcov={args.lambda_cov} λuni={args.lambda_uni}  "
          f"γ={args.vic_gamma}  hnm_every={args.hnm_every}")
    print("=" * 78)

    for ep in range(1, args.epochs + 1):
        if ep == 1 or (ep % args.hnm_every == 0):
            print(f"  [HNM] rebuilding index at ep {ep}...", flush=True)
            hard_negs = H.build_hnm_index(encoder, train_segs, train_wl, device,
                                          target_length=args.target_length)
            encoder.train()

        encoder.train()
        ep_loss = ep_nt = ep_var = ep_cov = ep_uni = 0.0
        t0 = time.time()

        for _ in range(args.steps):
            optimizer.zero_grad()
            n_random = args.ww_batch // 2
            n_hard   = args.ww_batch - n_random
            rand_words = random.sample(train_wl, min(n_random, len(train_wl)))
            hard_words = []
            if hard_negs:
                for w in rand_words[:n_hard]:
                    neg_pool = hard_negs.get(w, [])
                    if neg_pool:
                        hw = random.choice(neg_pool[:10])
                        if hw in train_segs:
                            hard_words.append(hw)
            ww_words = list(dict.fromkeys(rand_words + hard_words))[:args.ww_batch]

            aa, bb = [], []
            for w in ww_words:
                a, b = H._pick2(train_segs[w])
                aa.append(H.augment(a, w, directional, args.flip_prob,
                                    args.warp_prob, warp_range, args.noise_std))
                bb.append(H.augment(b, w, directional, args.flip_prob,
                                    args.warp_prob, warp_range, args.noise_std))
            ta = torch.from_numpy(np.stack(aa)).float().to(device)
            tb = torch.from_numpy(np.stack(bb)).float().to(device)

            na, pa = encoder(ta, return_prenorm=True)   # (normed, prenorm)
            nb, pb = encoder(tb, return_prenorm=True)

            nt = H.nt_xent(na, nb, args.temperature)
            var_a, cov_a = vicreg_var_cov(pa, gamma=args.vic_gamma)
            var_b, cov_b = vicreg_var_cov(pb, gamma=args.vic_gamma)
            var = 0.5 * (var_a + var_b)
            cov = 0.5 * (cov_a + cov_b)
            uni = (0.5 * (uniformity_loss(na) + uniformity_loss(nb))
                   if args.lambda_uni > 0 else torch.zeros((), device=device))

            loss = nt + args.lambda_var * var + args.lambda_cov * cov + args.lambda_uni * uni
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step(); scheduler.step()
            ep_loss += loss.item(); ep_nt += nt.item()
            ep_var += var.item(); ep_cov += cov.item(); ep_uni += float(uni.detach())

        # val loss (NT-Xent 만, 비교 가능하게)
        encoder.eval()
        with torch.no_grad():
            vl = 0.0
            for _ in range(args.val_steps):
                ww = random.sample(val_wl, min(args.ww_batch, len(val_wl)))
                av, bv = [], []
                for w in ww:
                    a, b = H._pick2(val_segs[w]); av.append(a); bv.append(b)
                ta = torch.from_numpy(np.stack(av)).float().to(device)
                tb = torch.from_numpy(np.stack(bv)).float().to(device)
                vl += H.nt_xent(encoder(ta), encoder(tb), args.temperature).item()
            val_loss = vl / args.val_steps

        elapsed = time.time() - t0
        s = args.steps
        print(f"Ep {ep:03d}/{args.epochs}  loss={ep_loss/s:.4f}  nt={ep_nt/s:.4f}  "
              f"var={ep_var/s:.4f}  cov={ep_cov/s:.4f}  uni={ep_uni/s:.4f}  "
              f"val={val_loss:.4f}  {elapsed:.0f}s", flush=True)

        if ep % args.val_every == 0:
            rk = measure_rank(encoder, val_segs, device)
            ext = H.eval_loo(encoder, recorded, device)
            marker = ""
            if ext["top3"] > best_rec_top3:
                best_rec_top3 = ext["top3"]
                torch.save({k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                           os.path.join(args.output, CKPT_NAME))
                patience_count = 0; marker = " ★"
            else:
                patience_count += 1
            print(f"  [rank] PR={rk['participation_ratio']:.1f}/{rk['dim']}  "
                  f"entropy_rank={rk['entropy_rank']:.1f}  top1_var={rk['top1_var_frac']:.1%}",
                  flush=True)
            print(f"  [LOO] Top-1:{ext['top1']:.1%}  Top-3:{ext['top3']:.1%}  (n={ext['total']}){marker}",
                  flush=True)
            history.append({"ep": ep, "loss": round(ep_loss/s, 4), "nt": round(ep_nt/s, 4),
                            "var": round(ep_var/s, 4), "cov": round(ep_cov/s, 4),
                            "val": round(val_loss, 4), "rec_top3": round(ext["top3"], 4),
                            "pr": round(rk["participation_ratio"], 2),
                            "entropy_rank": round(rk["entropy_rank"], 2)})
            json.dump(history, open(os.path.join(args.output, "history.json"), "w"), indent=2)
            if patience_count >= args.patience:
                print(f"[early stop] patience={args.patience} at ep {ep}"); break

    print("=" * 78)
    print(f"[done] best LOO Top-3 = {best_rec_top3:.4f}")
    json.dump({"best_rec_top3": best_rec_top3},
              open(os.path.join(args.output, "results.json"), "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--aihub-dir",    default=AIHUB_DIR)
    p.add_argument("--output",       default=OUT_DIR)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--steps",        type=int,   default=500)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--temperature",  type=float, default=0.07)
    p.add_argument("--ww-batch",     type=int,   default=64)
    p.add_argument("--val-every",    type=int,   default=5)
    p.add_argument("--patience",     type=int,   default=8)
    p.add_argument("--val-steps",    type=int,   default=30)
    p.add_argument("--hnm-every",    type=int,   default=5)
    # ── collapse 방지 항 ──────────────────────────────────────────────────────
    p.add_argument("--lambda-var",   type=float, default=1.0, help="VICReg variance hinge 가중치")
    p.add_argument("--lambda-cov",   type=float, default=0.04, help="VICReg covariance 가중치")
    p.add_argument("--lambda-uni",   type=float, default=0.0, help="uniformity(정규화後) 가중치(선택)")
    p.add_argument("--vic-gamma",    type=float, default=1.0, help="variance hinge 목표 std")
    # ── 공통 ─────────────────────────────────────────────────────────────────
    p.add_argument("--recorded-dir", default=RECORDED_DIR)
    p.add_argument("--target-length",type=int,   default=TARGET_LENGTH)
    p.add_argument("--fps-norm",     action="store_true")
    p.add_argument("--target-fps",   type=float, default=15.0)
    p.add_argument("--d-model",      type=int,   default=256)
    p.add_argument("--nhead",        type=int,   default=8)
    p.add_argument("--num-layers",   type=int,   default=2)
    p.add_argument("--dim-ff",       type=int,   default=512)
    p.add_argument("--dropout",      type=float, default=0.1)
    p.add_argument("--wd",           type=float, default=1e-4)
    p.add_argument("--warp-prob",    type=float, default=0.8)
    p.add_argument("--warp-min",     type=float, default=0.5)
    p.add_argument("--warp-max",     type=float, default=2.0)
    p.add_argument("--noise-std",    type=float, default=0.005)
    p.add_argument("--flip-prob",    type=float, default=0.5)
    args = p.parse_args()
    train(args)
