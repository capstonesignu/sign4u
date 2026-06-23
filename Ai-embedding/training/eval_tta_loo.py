"""
eval_tta_loo.py — Test-Time Augmentation LOO 평가.

훈련 데이터를 전혀 건드리지 않고, 추론 시 각 recorded-video 샘플을
N번 augment → 평균 임베딩으로 LOO 평가.

정당성: TTA는 eval procedure 변경일 뿐, 학습 데이터에 recorded-video
        를 포함하지 않음.

Usage:
    python eval_tta_loo.py                        # Phase 1 best
    python eval_tta_loo.py --model result_cs_30fps_t64/encoder_cross_stream_best.pt
    python eval_tta_loo.py --tta 20 --noise 0.01
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 폴더의 model/preprocess/services

from model import LandmarkCrossStreamEncoder, LandmarkStreamEncoder
from preprocess import temporal_interpolate, TARGET_LENGTH
from services.preprocessing import convert_stream

PRESET       = "B"
RECORDED_DIR = "dataset/recorded-video"
DEFAULT_MODEL = "result_cross_stream_tw/encoder_cross_stream_best.pt"
SEED         = 42


def time_warp(seq: np.ndarray, rate_range=(0.7, 1.3)) -> np.ndarray:
    T, _ = seq.shape
    rate = random.uniform(*rate_range)
    new_len = max(8, int(round(T * rate)))
    idx = np.linspace(0, T - 1, new_len)
    w = seq[np.round(idx).astype(int)]
    b = np.linspace(0, new_len - 1, T)
    return w[np.round(b).astype(int)].astype(seq.dtype)


def augment(seq: np.ndarray, noise_std=0.005, warp_prob=0.7,
            warp_range=(0.8, 1.2)) -> np.ndarray:
    out = seq.copy()
    if random.random() < warp_prob:
        out = time_warp(out, warp_range)
    out += np.random.normal(0.0, noise_std, out.shape).astype(out.dtype)
    return out


def load_recorded(recorded_dir: str,
                  target_length: int = TARGET_LENGTH,
                  fps_norm: bool = False,
                  target_fps: float = 30.0) -> Dict[str, List[np.ndarray]]:
    segs: Dict[str, List[np.ndarray]] = {}
    for wd in sorted(Path(recorded_dir).iterdir()):
        if not wd.is_dir(): continue
        samples = []
        for npz in sorted(wd.glob("*.npz")):
            try:
                data = np.load(str(npz))
                T    = data["pose"].shape[0]
                kps  = np.concatenate([
                    data["pose"].reshape(T, -1),
                    data["left_hand"].reshape(T, -1),
                    data["right_hand"].reshape(T, -1),
                    data["face"].reshape(T, -1),
                ], axis=-1)
                seg = convert_stream(kps, target_preset=PRESET)
                if fps_norm and "fps" in data:
                    src_fps  = float(data["fps"])
                    norm_len = max(8, int(round(len(seg) * target_fps / src_fps)))
                    seg = temporal_interpolate(seg, norm_len)
                seg = temporal_interpolate(seg, target_length)
                samples.append(seg.astype(np.float32))
            except Exception:
                pass
        if samples:
            segs[wd.name] = samples
    return segs


@torch.no_grad()
def embed_with_tta(encoder, seq: np.ndarray, device,
                   n_tta: int = 1,
                   noise_std: float = 0.005,
                   warp_range=(0.8, 1.2)) -> np.ndarray:
    """단일 샘플 → TTA 평균 임베딩."""
    embs = []
    # 원본 임베딩 항상 포함
    x = torch.from_numpy(seq).unsqueeze(0).float().to(device)
    e = encoder(x)
    embs.append(F.normalize(e[0], dim=0).cpu().numpy())
    # N번 augmented 버전
    for _ in range(n_tta - 1):
        aug = augment(seq, noise_std=noise_std, warp_range=warp_range)
        x   = torch.from_numpy(aug).unsqueeze(0).float().to(device)
        e   = encoder(x)
        embs.append(F.normalize(e[0], dim=0).cpu().numpy())
    avg = np.mean(embs, axis=0).astype(np.float32)
    avg /= (np.linalg.norm(avg) + 1e-8)
    return avg


def eval_loo(encoder, recorded: Dict[str, List[np.ndarray]], device,
             n_tta: int = 1, noise_std: float = 0.005,
             warp_range=(0.8, 1.2)) -> Dict[str, float]:
    encoder.eval()
    word_embs: Dict[str, List[np.ndarray]] = {}
    for word, segs in recorded.items():
        word_embs[word] = [
            embed_with_tta(encoder, s, device, n_tta, noise_std, warp_range)
            for s in segs
        ]

    top1 = top3 = total = 0
    for qw, embs in word_embs.items():
        for i, qe in enumerate(embs):
            db_v, db_l = [], []
            for w, wembs in word_embs.items():
                for j, e in enumerate(wembs):
                    if w == qw and j == i: continue
                    db_v.append(e); db_l.append(w)
            if not db_v: continue
            mat = np.stack(db_v).astype(np.float32); faiss.normalize_L2(mat)
            fidx = faiss.IndexFlatIP(mat.shape[1]); fidx.add(mat)
            q = qe.reshape(1, -1).copy(); faiss.normalize_L2(q)
            _, I = fidx.search(q, min(3, len(db_l)))
            top3w = [db_l[ii] for ii in I[0]]
            if top3w[0] == qw: top1 += 1
            if qw in top3w:    top3 += 1
            total += 1
    return {"top1": top1/total, "top3": top3/total, "total": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default=DEFAULT_MODEL)
    p.add_argument("--recorded-dir", default=RECORDED_DIR)
    p.add_argument("--target-length",type=int,   default=TARGET_LENGTH)
    p.add_argument("--fps-norm",     action="store_true")
    p.add_argument("--target-fps",   type=float, default=30.0)
    p.add_argument("--tta",          type=int,   default=10,
                   help="TTA 횟수 (1=TTA 없음, 10=10번 aug 평균)")
    p.add_argument("--noise",        type=float, default=0.005)
    p.add_argument("--warp-min",     type=float, default=0.8)
    p.add_argument("--warp-max",     type=float, default=1.2)
    p.add_argument("--d-model",      type=int,   default=256)
    p.add_argument("--nhead",        type=int,   default=8)
    p.add_argument("--num-layers",   type=int,   default=2)
    p.add_argument("--dim-ff",       type=int,   default=512)
    p.add_argument("--model-type",   default="cross_stream",
                   choices=["cross_stream", "stream"])
    args = p.parse_args()

    random.seed(SEED); np.random.seed(SEED)

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))

    ModelCls = LandmarkCrossStreamEncoder if args.model_type == "cross_stream" else LandmarkStreamEncoder
    encoder = ModelCls(
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_ff,
        dropout=0.0, norm_first=True,
    ).to(device)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    encoder.load_state_dict(state, strict=True)
    print(f"[model] {args.model}")

    recorded = load_recorded(
        args.recorded_dir,
        target_length=args.target_length,
        fps_norm=args.fps_norm,
        target_fps=args.target_fps,
    )
    print(f"[recorded] {len(recorded)} words  "
          f"{sum(len(v) for v in recorded.values())} samples  "
          f"target_length={args.target_length}  fps_norm={args.fps_norm}")

    warp_range = (args.warp_min, args.warp_max)

    # TTA=1 (baseline)
    r1 = eval_loo(encoder, recorded, device, n_tta=1)
    print(f"\nTTA=1  (no aug)   Top-1: {r1['top1']:.1%}  Top-3: {r1['top3']:.1%}")

    # TTA=N
    if args.tta > 1:
        rn = eval_loo(encoder, recorded, device,
                      n_tta=args.tta, noise_std=args.noise,
                      warp_range=warp_range)
        print(f"TTA={args.tta:<2d}            Top-1: {rn['top1']:.1%}  "
              f"Top-3: {rn['top3']:.1%}  "
              f"(noise={args.noise}  warp={warp_range})")


if __name__ == "__main__":
    main()
