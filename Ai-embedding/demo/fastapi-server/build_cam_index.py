"""
build_cam_index.py

Camera-aware 인코더 + AI Hub 전체 word-crop (camera_occlusion=True) 으로 FAISS 빌드.

DB 빌드 시 camera_occlusion=True 를 적용하는 것이 핵심:
  실제 데모 카메라가 shoulder-hip 중점 아래를 볼 수 없으므로,
  DB 임베딩도 동일한 전처리(오클루전 → 보간)를 거쳐야
  쿼리(실시간 카메라)와 동일한 feature space에 놓인다.

출력:
  index/cam_wordcrop.faiss
  index/cam_wordcrop.faiss.labels.json

Usage:
    cd demo/fastapi-server
    python build_cam_index.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import defaultdict
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
from preprocess import feature_dim_for, temporal_interpolate, TARGET_LENGTH, _load_npz_raw

PRESET       = "B"
WORD_KP_DIR  = PROJECT_ROOT / "dataset" / "word-keypoints" / "1.Training"
WORD_MAP     = PROJECT_ROOT / "word_mapping.json"
CAM_MODEL    = PROJECT_ROOT / "result_word_encoder_cam" / "encoder_cam_best.pt"
OUT_DIR      = Path(__file__).resolve().parent / "index"

_WORD_FILE_RE = re.compile(r"NIA_SL_(WORD\d+)_")


def load_encoder() -> SignEncoder:
    ckpt  = torch.load(str(CAM_MODEL), map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    enc   = SignEncoder(
        input_dim=feature_dim_for(PRESET),
        d_model=256, nhead=8, num_layers=2,
        dim_feedforward=1024, dropout=0.0, norm_first=True,
    )
    enc.load_state_dict(state)
    enc.eval()
    print(f"[encoder] loaded {CAM_MODEL}")
    return enc


def load_word_crops_cam(word_map: dict) -> tuple[dict, int]:
    """AI Hub word-crop 로드 + camera_occlusion=True 적용."""
    segs: dict[str, list[np.ndarray]] = defaultdict(list)
    skipped = 0
    for path in glob.glob(str(WORD_KP_DIR / "**" / "*.npz"), recursive=True):
        m = _WORD_FILE_RE.search(os.path.basename(path))
        if not m:
            continue
        kor = word_map.get(m.group(1))
        if not kor:
            continue
        try:
            d      = np.load(path, allow_pickle=True)
            stream = _load_npz_raw(path, PRESET, camera_occlusion=True)
            if "start_sec" in d and "end_sec" in d:
                fps  = float(d["fps"]); T = stream.shape[0]
                sf   = max(0, round(float(d["start_sec"]) * fps))
                ef   = min(T - 1, round(float(d["end_sec"]) * fps))
                crop = stream[sf: ef + 1] if ef > sf + 1 else stream
            else:
                crop = stream
            if len(crop) < 3:
                skipped += 1
                continue
            segs[kor].append(temporal_interpolate(crop, TARGET_LENGTH).astype(np.float32))
        except Exception:
            skipped += 1
    total = sum(len(v) for v in segs.values())
    print(f"[data] {len(segs)} words, {total} segs (skipped {skipped})")
    return dict(segs), total


@torch.no_grad()
def embed_all(encoder: SignEncoder, segs: dict, batch: int = 512) -> tuple[np.ndarray, list[str]]:
    device = torch.device("cpu")
    encoder.to(device)
    vecs, lbls = [], []
    words = sorted(segs.keys())
    for wi, w in enumerate(words):
        for i in range(0, len(segs[w]), batch):
            chunk = segs[w][i: i + batch]
            x = torch.from_numpy(np.stack(chunk)).float().to(device)
            e = F.normalize(encoder(x), dim=1).cpu().numpy()
            vecs.append(e)
            lbls.extend([w] * len(chunk))
        if (wi + 1) % 500 == 0:
            print(f"  {wi + 1}/{len(words)} words embedded...", flush=True)
    mat = np.concatenate(vecs, axis=0).astype(np.float32)
    print(f"[embed] {mat.shape[0]} vectors, dim={mat.shape[1]}")
    return mat, lbls


def main():
    OUT_DIR.mkdir(exist_ok=True)
    word_map: dict = json.load(open(WORD_MAP))

    print("[1] AI Hub word-crop 로딩 (camera_occlusion=True)...")
    segs, _ = load_word_crops_cam(word_map)

    print("[2] Camera-aware 인코더 로딩...")
    encoder = load_encoder()

    print("[3] 임베딩 중...")
    mat, lbls = embed_all(encoder, segs)

    print("[4] FAISS 빌드 & 저장...")
    faiss.normalize_L2(mat)
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)

    out_faiss  = OUT_DIR / "cam_wordcrop.faiss"
    out_labels = OUT_DIR / "cam_wordcrop.faiss.labels.json"
    faiss.write_index(idx, str(out_faiss))
    out_labels.write_text(json.dumps(lbls, ensure_ascii=False), encoding="utf-8")

    print(f"\n완료: {out_faiss}")
    print(f"  vectors={idx.ntotal}, words={len(set(lbls))}, dim={idx.d}")


if __name__ == "__main__":
    main()
