"""
build_recorded_index.py

dataset/recorded-video 단어만으로 FAISS 인덱스 빌드.
30fps Fixed Baseline 인코더 사용 (preset B, T=64).

각 샘플을 원본 + 좌우 반전 2개로 넣어 다양성 확보.

Usage:
    cd demo/fastapi-server
    python build_recorded_index.py
    python build_recorded_index.py --output index/recorded_30fps.faiss
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import faiss
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import SignEncoder
from preprocess import feature_dim_for, temporal_interpolate, _load_npz_raw

PRESET       = "B"
TARGET_LEN   = 64
MODEL_PATH   = PROJECT_ROOT / "result_30fps_fixed_baseline" / "encoder_30fps_fixed_baseline_best.pt"
RECORDED_DIR = PROJECT_ROOT / "dataset" / "recorded-video"
OUT_DIR      = Path(__file__).resolve().parent / "index"


def flip_horizontal_B(seq: np.ndarray) -> np.ndarray:
    """Preset B (T, 136) 좌우 반전 — train_30fps_fixed.py와 동일."""
    s = seq.copy()
    s[:, 0::2] *= -1
    for l_start, r_start in [(2, 4), (6, 8), (10, 12)]:
        tmp = s[:, l_start:l_start + 2].copy()
        s[:, l_start:l_start + 2] = s[:, r_start:r_start + 2]
        s[:, r_start:r_start + 2] = tmp
    tmp = s[:, 14:56].copy()
    s[:, 14:56] = s[:, 56:98]
    s[:, 56:98] = tmp
    return s


def load_encoder() -> SignEncoder:
    ckpt = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=True)
    state = ckpt.get("encoder_state_dict", ckpt)
    enc = SignEncoder(
        input_dim=feature_dim_for(PRESET),
        d_model=256, nhead=8, num_layers=4,
        dim_feedforward=1024, dropout=0.0, norm_first=True,
    )
    enc.load_state_dict(state)
    enc.eval()
    print(f"[encoder] {MODEL_PATH}")
    return enc


def load_recorded(recorded_dir: Path) -> dict[str, list[np.ndarray]]:
    """recorded-video/{label}/*.npz → 원본 + 좌우반전 2배."""
    segs: dict[str, list[np.ndarray]] = {}
    skipped = 0

    for word_dir in sorted(recorded_dir.iterdir()):
        if not word_dir.is_dir():
            continue
        label = word_dir.name
        npzs = sorted(word_dir.glob("*.npz"))
        if not npzs:
            continue

        samples = []
        for npz_path in npzs:
            try:
                raw = _load_npz_raw(str(npz_path), PRESET)
                if len(raw) < 3:
                    skipped += 1
                    continue
                fixed = temporal_interpolate(raw, TARGET_LEN).astype(np.float32)
                flipped = flip_horizontal_B(fixed)
                samples.append(fixed)
                samples.append(flipped)
            except Exception as e:
                print(f"  [skip] {npz_path.name}: {e}")
                skipped += 1
                continue

        if samples:
            segs[label] = samples

    total = sum(len(v) for v in segs.values())
    print(f"[data] {len(segs)} words, {total} samples "
          f"(원본+반전 각 {total//2}, skipped {skipped})")
    return segs


@torch.no_grad()
def embed_all(encoder: SignEncoder, segs: dict[str, list[np.ndarray]],
              batch: int = 256) -> tuple[np.ndarray, list[str]]:
    vecs, lbls = [], []
    for label, samples in sorted(segs.items()):
        for i in range(0, len(samples), batch):
            chunk = samples[i:i + batch]
            x = torch.from_numpy(np.stack(chunk)).float()
            e = F.normalize(encoder(x), dim=1).cpu().numpy()
            vecs.append(e)
            lbls.extend([label] * len(chunk))
    mat = np.concatenate(vecs, axis=0).astype(np.float32)
    print(f"[embed] {mat.shape[0]} vectors, dim={mat.shape[1]}")
    return mat, lbls


def main(output: str):
    OUT_DIR.mkdir(exist_ok=True)

    print("[1] recorded-video 로딩 (원본 + 좌우반전)...")
    segs = load_recorded(RECORDED_DIR)

    print("[2] 인코더 로딩...")
    encoder = load_encoder()

    print("[3] 임베딩 중...")
    mat, lbls = embed_all(encoder, segs)

    print("[4] FAISS 빌드 & 저장...")
    faiss.normalize_L2(mat)
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)

    out_faiss  = Path(output)
    out_labels = Path(str(output) + ".labels.json")
    faiss.write_index(idx, str(out_faiss))
    out_labels.write_text(json.dumps(lbls, ensure_ascii=False), encoding="utf-8")

    print(f"\n완료: {out_faiss}")
    print(f"  vectors={idx.ntotal}, words={len(set(lbls))}, dim={idx.d}")
    for label in sorted(set(lbls)):
        cnt = lbls.count(label)
        print(f"  {label}: {cnt}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUT_DIR / "recorded_30fps.faiss"))
    args = parser.parse_args()
    main(args.output)
