"""Build FAISS index from the trained model and aihub dataset.

Usage:
    python build_index.py
    python build_index.py --model /path/to/best_model.pt --preset B --seq-len 128
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import faiss
from model import SignEncoder
from data import load_dataset
from preprocess import load_npz_keypoints, feature_dim_for

import config


def build(model_path: str, dataset_path: str, preset: str,
          seq_len: int, output_path: str, word_mapping_path: str,
          extra_dataset_paths: list | None = None):
    extra_dataset_paths = extra_dataset_paths or []
    print(f"[build] Model: {model_path}")
    print(f"[build] Dataset: {dataset_path}")
    if extra_dataset_paths:
        print(f"[build] Extra datasets: {extra_dataset_paths}")
    print(f"[build] Preset: {preset}, seq_len: {seq_len}")

    input_dim = feature_dim_for(preset)
    print(f"[build] input_dim: {input_dim}")

    encoder = SignEncoder(input_dim=input_dim)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if "encoder_state_dict" in state:
        state = state["encoder_state_dict"]
    encoder.load_state_dict(state)
    encoder.eval()

    print("[build] Loading dataset...")
    raw_dataset = load_dataset(dataset_path)
    print(f"[build] Found {len(raw_dataset)} words in primary dataset")

    for extra_path in extra_dataset_paths:
        print(f"[build] Loading extra dataset: {extra_path}")
        extra = load_dataset(extra_path)
        for word_id, signer_map in extra.items():
            tgt = raw_dataset.setdefault(word_id, {})
            for signer_id, paths in signer_map.items():
                tgt.setdefault(signer_id, []).extend(paths)
        print(f"[build] After merge: {len(raw_dataset)} words total")

    raw_dataset = {k: v for k, v in raw_dataset.items() if not k.startswith("FS")}
    print(f"[build] After excluding fingerspelling (FS): {len(raw_dataset)} words")

    word_map = {}
    if os.path.exists(word_mapping_path):
        with open(word_mapping_path, "r", encoding="utf-8") as f:
            word_map = json.load(f)

    # 단어별 embedding 수집
    word_embs: dict = {}
    skipped = 0

    for i, (word_id, signer_dict) in enumerate(sorted(raw_dataset.items())):
        samples = []
        for signer_id, paths in signer_dict.items():
            for path in paths:
                if not path.endswith(".npz"):
                    continue
                try:
                    kp = load_npz_keypoints(path, preset, seq_len)
                    samples.append(kp)
                except Exception:
                    skipped += 1
                    continue

        if not samples:
            continue

        batch = torch.from_numpy(np.stack(samples))
        with torch.no_grad():
            embs = encoder(batch).cpu().numpy().astype(np.float32)
        word_embs[word_id] = embs

        if (i + 1) % 50 == 0:
            korean = word_map.get(word_id, "")
            print(f"  [{i+1}/{len(raw_dataset)}] {word_id} ({korean}): "
                  f"{len(samples)} samples")

    if not word_embs:
        print("[build] ERROR: No embeddings generated.")
        return

    # 단어당 1개 prototype (평균 벡터, L2 정규화)
    all_embs = []
    all_labels = []
    for word_id, embs in sorted(word_embs.items()):
        proto = embs.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        all_embs.append(proto)
        all_labels.append(word_id)

    embeddings = np.vstack(all_embs).astype(np.float32)
    print(f"[build] Prototypes: {embeddings.shape[0]} words "
          f"(skipped {skipped} files)")
    print(f"[build] Embedding dim: {embeddings.shape[1]}")

    # 3000개 미만이므로 IndexFlatL2 (exact search)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print(f"[build] IndexFlatL2: {index.ntotal} vectors")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, output_path)
    labels_path = output_path + ".labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, ensure_ascii=False)

    print(f"[build] Saved: {output_path}")
    print(f"[build] Saved: {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.EMBEDDING_MODEL_PATH)
    parser.add_argument("--dataset", default=config.DATASET_PATH)
    parser.add_argument("--preset", default=config.FEATURE_PRESET)
    parser.add_argument("--seq-len", type=int, default=config.SEQUENCE_LENGTH)
    parser.add_argument("--output", default=config.FAISS_INDEX_PATH)
    parser.add_argument("--word-mapping", default=config.WORD_MAPPING_PATH)
    parser.add_argument("--extra-datasets", nargs="*",
                        default=config.EXTRA_DATASET_PATHS,
                        help="추가 데이터셋 경로 (검증 농인 샘플 포함 등)")
    args = parser.parse_args()

    build(args.model, args.dataset, args.preset, args.seq_len,
          args.output, args.word_mapping, args.extra_datasets)
