"""
build_jbedu_db.py — jbedu-recordings FAISS DB 재구축 (flip-free, 인코더 선택).

upload_to_hf.py 와 동일 전처리(convert_stream B → 15fps norm → 64프레임 → L2)지만:
  - flip_horizontal_B 제거 (no-mirror 하드룰 준수; Phase 1 에서 flip 벡터가 같은 단어를
    원본과 거의 직교로 만들고 cross-word 충돌을 부풀림을 확인).
  - 인코더 경로를 인자로 받음 (기본: collapse-fix ep50).
  - 출력은 staging 디렉토리(기본 db_staging/) — 실 db/ 를 건드리지 않음(배포 전 검증용).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
        python build_jbedu_db.py
    python build_jbedu_db.py --encoder result_cross_stream_tw/encoder_cross_stream_best.pt \
        --out-dir db_staging_old --keep-flip   # 비교용(구 인코더 + flip)
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
import torch
import torch.nn.functional as F

import upload_to_hf as U   # load_npz / flip_horizontal_B / 전처리 상수 재사용
from upload_to_hf import RECORDINGS_DIR, FAISS_NAME, LandmarkCrossStreamEncoder


def load_recordings_flexible(rec_dir: Path, keep_flip: bool):
    segs: dict[str, list[np.ndarray]] = {}
    for word_dir in sorted(rec_dir.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("."):
            continue
        samples = []
        for npz in sorted(word_dir.glob("*.npz")):
            seg = U.load_npz(npz)
            if seg is not None:
                samples.append(seg)
                if keep_flip:
                    samples.append(U.flip_horizontal_B(seg))
        if samples:
            segs[word_dir.name] = samples
    total = sum(len(v) for v in segs.values())
    print(f"[data] {len(segs)} 단어, {total} 벡터 (flip={'포함' if keep_flip else '제외'})")
    return segs


def load_encoder(path: str, device):
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    enc = LandmarkCrossStreamEncoder(d_model=256, nhead=8, num_layers=2,
                                     dim_feedforward=512, dropout=0.0, norm_first=True).to(device)
    enc.load_state_dict(state, strict=True); enc.eval()
    w = F.softmax(enc.stream_logits, dim=0).detach().cpu().numpy()
    print(f"[encoder] {path}  stream→pose={w[0]:.3f} L={w[1]:.3f} R={w[2]:.3f} face={w[3]:.3f}")
    return enc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="result_cs_collapse/encoder_cross_stream_best.pt")
    p.add_argument("--out-dir", default="db_staging")
    p.add_argument("--keep-flip", action="store_true", help="좌우반전 벡터도 추가(비교용)")
    args = p.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[device] {device}")

    segs = load_recordings_flexible(RECORDINGS_DIR, args.keep_flip)
    encoder = load_encoder(args.encoder, device)
    mat, lbls = U.embed_all(encoder, segs, device)

    faiss.normalize_L2(mat)
    idx = faiss.IndexFlatIP(mat.shape[1]); idx.add(mat)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fp = out / f"{FAISS_NAME}.faiss"
    lp = out / f"{FAISS_NAME}.faiss.labels.json"
    faiss.write_index(idx, str(fp))
    lp.write_text(json.dumps(lbls, ensure_ascii=False), encoding="utf-8")
    print(f"[faiss] {fp}  vectors={idx.ntotal}  words={len(set(lbls))}  dim={idx.d}")


if __name__ == "__main__":
    main()
