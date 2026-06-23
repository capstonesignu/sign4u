"""
build_exp026_index.py

EXP-026 인코더 + full-vocab FAISS 빌드 스크립트.

Word proto (2500) + OOV sentence proto (368) 합쳐서 저장:
  index/exp026_fullvocab.faiss
  index/exp026_fullvocab.faiss.labels.json

Usage:
    cd demo/fastapi-server
    python build_exp026_index.py
"""
from __future__ import annotations

import json
import os
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
os.chdir(PROJECT_ROOT)  # train_global_proto 상대 경로 기준

from model import SignEncoder
from preprocess import feature_dim_for
from train_global_proto import (
    PRESET, VAR_LEN_MAX,
    _load_npz_raw, load_split_varlen, make_padded_batch,
    temporal_interpolate, build_kw_to_wid_map, build_wid_to_kw_map,
)

EXP026_ENCODER  = PROJECT_ROOT / "result_openvocab/exp026_s3ood/global_proto_best.pt"
MORPH_DIR       = PROJECT_ROOT / "dataset/sentence-keypoints/morpheme"
SEN_TRAIN_ROOT  = PROJECT_ROOT / "dataset/sentence-keypoints/1.Training"
SEN_VAL_ROOT    = PROJECT_ROOT / "dataset/sentence-keypoints/2.Validation"
OUT_DIR         = Path(__file__).resolve().parent / "index"
HELD_OUT_START  = 2501
MAX_PROTO       = 16   # OOV 단어당 최대 인스턴스 수


def load_encoder() -> SignEncoder:
    ckpt = torch.load(str(EXP026_ENCODER), map_location="cpu")
    enc = SignEncoder(
        input_dim       = ckpt.get("input_dim", feature_dim_for(PRESET)),
        d_model         = ckpt.get("d_model", 256),
        nhead           = 8,
        num_layers      = 2,
        dim_feedforward = 1024,
        dropout         = 0.0,
        norm_first      = True,
    )
    enc.load_state_dict(ckpt["encoder_state_dict"])
    enc.eval()
    return enc


@torch.no_grad()
def build_word_protos(encoder, train_ds, n_proto=8, chunk=256):
    # 동음이의어 포함 전체 직접 매핑
    wid_to_kw = build_wid_to_kw_map(valid_wid_set=set(train_ds.keys()))

    wid_list = sorted(train_ds.keys())
    all_segs, all_wi = [], []
    for wi, wid in enumerate(wid_list):
        for s in train_ds[wid][:n_proto]:
            all_segs.append(s)
            all_wi.append(wi)

    accum = [[] for _ in range(len(wid_list))]
    for start in range(0, len(all_segs), chunk):
        cs = all_segs[start:start + chunk]
        cw = all_wi[start:start + chunk]
        x, mask = make_padded_batch(cs)
        embs = encoder(x, src_key_padding_mask=mask)
        for i, wi in enumerate(cw):
            accum[wi].append(embs[i].cpu())

    mats, labels = [], []
    for wi, wid in enumerate(wid_list):
        emb = torch.stack(accum[wi]).mean(0)
        mats.append(F.normalize(emb, dim=0).numpy().astype("float32"))
        labels.append(wid_to_kw.get(wid, wid))

    mapped = sum(1 for l in labels if not l.startswith("WORD"))
    print(f"  Word proto: {len(labels)}개, 한국어 매핑={mapped}/{len(labels)}")
    return np.stack(mats), labels


def build_npz_index():
    idx = {}
    for root in [SEN_TRAIN_ROOT, SEN_VAL_ROOT]:
        for p in Path(root).rglob("*.npz"):
            idx[p.stem] = p
    return idx


def collect_oov_segs(oov_words, npz_index, fps=15.0):
    segs = defaultdict(list)
    need = {w: MAX_PROTO for w in oov_words}

    for morph in sorted(MORPH_DIR.rglob("*_morpheme.json")):
        if not need:
            break
        stem = morph.stem.replace("_morpheme", "")
        npz_path = npz_index.get(stem)
        if npz_path is None:
            continue

        d = json.loads(morph.read_text(encoding="utf-8"))
        hits = []
        for seg in d.get("data", []):
            attrs = seg.get("attributes", [])
            if not attrs:
                continue
            name = attrs[0].get("name", "").strip()
            if name in need and need[name] > 0:
                sf = round(float(seg["start"]) * fps)
                ef = round(float(seg["end"])   * fps)
                hits.append((name, sf, ef))

        if not hits:
            continue

        stream = _load_npz_raw(str(npz_path), PRESET)
        if stream is None or len(stream) < 4:
            continue
        T = len(stream)

        for name, sf, ef in hits:
            if need.get(name, 0) <= 0:
                continue
            sf = max(0, sf); ef = min(T - 1, ef)
            seg_arr = stream[sf:ef + 1]
            if len(seg_arr) < 2:
                seg_arr = stream[max(0, sf - 1):ef + 2]
            if len(seg_arr) > VAR_LEN_MAX:
                seg_arr = temporal_interpolate(seg_arr, VAR_LEN_MAX)
            if len(seg_arr) < 2:
                continue
            segs[name].append(seg_arr.astype(np.float32))
            need[name] -= 1
            if need[name] <= 0:
                del need[name]

    return dict(segs)


@torch.no_grad()
def build_oov_protos(encoder, oov_segs, chunk=256):
    words = sorted(oov_segs.keys())
    all_segs, all_wi = [], []
    for wi, w in enumerate(words):
        for s in oov_segs[w]:
            all_segs.append(s)
            all_wi.append(wi)

    accum = [[] for _ in range(len(words))]
    for start in range(0, len(all_segs), chunk):
        cs = all_segs[start:start + chunk]
        cw = all_wi[start:start + chunk]
        x, mask = make_padded_batch(cs)
        embs = encoder(x, src_key_padding_mask=mask)
        for i, wi in enumerate(cw):
            accum[wi].append(embs[i].cpu())

    mats, labels = [], []
    for wi, w in enumerate(words):
        if not accum[wi]:
            continue
        emb = torch.stack(accum[wi]).mean(0)
        mats.append(F.normalize(emb, dim=0).numpy().astype("float32"))
        labels.append(w)

    print(f"  OOV proto: {len(labels)}개 단어")
    return (np.stack(mats) if mats else np.zeros((0, 256), dtype="float32")), labels


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("[1] Training data 로드...")
    train_ds, _ = load_split_varlen(held_out_start=HELD_OUT_START)

    print("[2] EXP-026 인코더 로드...")
    encoder = load_encoder()

    print("[3] Word prototype 빌드...")
    word_mat, word_labels = build_word_protos(encoder, train_ds)

    # train vocab의 한국어 단어 집합 (동음이의어 포함)
    wid_to_kw_all = build_wid_to_kw_map(valid_wid_set=set(train_ds.keys()))
    train_kws     = set(wid_to_kw_all.values())

    print("[4] 문장 NPZ 인덱스 빌드...")
    npz_index = build_npz_index()
    print(f"  NPZ {len(npz_index)}개")

    print("[5] OOV 단어 스캔...")
    all_sent_words: set = set()
    for morph in MORPH_DIR.rglob("*_morpheme.json"):
        d = json.loads(morph.read_text(encoding="utf-8"))
        for seg in d.get("data", []):
            attrs = seg.get("attributes", [])
            if attrs:
                name = attrs[0].get("name", "").strip()
                if name:
                    all_sent_words.add(name)
    oov_words = all_sent_words - train_kws
    print(f"  전체 문장 어휘={len(all_sent_words)}, OOV={len(oov_words)}")

    print("[6] OOV 세그먼트 수집 및 prototype 빌드...")
    oov_segs = collect_oov_segs(oov_words, npz_index)
    oov_mat, oov_labels = build_oov_protos(encoder, oov_segs)

    print("[7] FAISS 빌드 및 저장...")
    if oov_mat.shape[0] > 0:
        all_mat = np.concatenate([word_mat, oov_mat], axis=0)
    else:
        all_mat = word_mat
    all_labels = word_labels + oov_labels

    idx = faiss.IndexFlatIP(all_mat.shape[1])
    idx.add(all_mat)

    faiss_path  = OUT_DIR / "exp026_fullvocab.faiss"
    labels_path = OUT_DIR / "exp026_fullvocab.faiss.labels.json"

    faiss.write_index(idx, str(faiss_path))
    labels_path.write_text(json.dumps(all_labels, ensure_ascii=False), encoding="utf-8")

    print(f"\n완료: {faiss_path}")
    print(f"  총 prototype: {idx.ntotal}개 (word={len(word_labels)}, oov={len(oov_labels)})")
    print(f"  labels: {labels_path}")


if __name__ == "__main__":
    main()
