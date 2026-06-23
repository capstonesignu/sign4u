"""
upload_to_hf.py — Phase 1 인코더 + jbedu-recordings FAISS 인덱스 빌드 후 HuggingFace 업로드.

Usage:
    huggingface-cli login          # 먼저 로그인
    python upload_to_hf.py --repo <username>/<repo-name>
    python upload_to_hf.py --repo Sogang-Capstone-2026-1/ksl-encoder
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import faiss
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 폴더의 services

from model import LandmarkCrossStreamEncoder
from preprocess import temporal_interpolate, TARGET_LENGTH
from services.preprocessing import convert_stream

PRESET         = "B"
TARGET_FPS     = 15.0
TARGET_LEN     = TARGET_LENGTH   # 64
ENCODER_PATH   = PROJECT_ROOT / "result_cross_stream_tw" / "encoder_cross_stream_best.pt"
# HF repo 경로: capstone/encoder/, capstone/db/ (기존 repo 구조 유지)
HF_ENCODER_PATH = "capstone/encoder/encoder_cross_stream_best.pt"
HF_FAISS_PREFIX = "capstone/db"
# 로컬 FAISS 빌드 경로
RECORDINGS_DIR = PROJECT_ROOT / "dataset" / "jbedu-recordings"
OUT_DIR        = PROJECT_ROOT / "db"
FAISS_NAME     = "jbedu_recordings"


# ── 좌우 반전 (Preset B) ───────────────────────────────────────────────────────
def flip_horizontal_B(seq: np.ndarray) -> np.ndarray:
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


# ── NPZ 로딩 (jbedu-recordings 포맷) ──────────────────────────────────────────
def load_npz(path: Path, fps_norm: bool = True) -> np.ndarray | None:
    try:
        data = np.load(str(path))
        T    = data["pose"].shape[0]
        kps  = np.concatenate([
            data["pose"].reshape(T, -1),
            data["left_hand"].reshape(T, -1),
            data["right_hand"].reshape(T, -1),
            data["face"].reshape(T, -1),
        ], axis=-1)                                   # (T, 210)
        seg = convert_stream(kps, target_preset=PRESET)   # (T, 136)
        if fps_norm and "fps" in data:
            src_fps  = float(data["fps"])
            norm_len = max(8, int(round(len(seg) * TARGET_FPS / src_fps)))
            seg = temporal_interpolate(seg, norm_len)
        return temporal_interpolate(seg, TARGET_LEN).astype(np.float32)
    except Exception as e:
        print(f"  [skip] {path.name}: {e}")
        return None


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def load_recordings(rec_dir: Path) -> dict[str, list[np.ndarray]]:
    segs: dict[str, list[np.ndarray]] = {}
    for word_dir in sorted(rec_dir.iterdir()):
        if not word_dir.is_dir() or word_dir.name.startswith("."):
            continue
        label = word_dir.name
        samples = []
        for npz in sorted(word_dir.glob("*.npz")):
            seg = load_npz(npz)
            if seg is not None:
                samples.append(seg)
                samples.append(flip_horizontal_B(seg))   # 좌우 반전 포함
        if samples:
            segs[label] = samples
    total = sum(len(v) for v in segs.values())
    print(f"[data] {len(segs)} 단어, {total} 벡터 (원본+반전)")
    return segs


# ── 인코더 로딩 ───────────────────────────────────────────────────────────────
def load_encoder(device: torch.device) -> LandmarkCrossStreamEncoder:
    ckpt  = torch.load(str(ENCODER_PATH), map_location="cpu", weights_only=False)
    state = ckpt.get("encoder_state_dict", ckpt)
    enc   = LandmarkCrossStreamEncoder(
        d_model=256, nhead=8, num_layers=2,
        dim_feedforward=512, dropout=0.0, norm_first=True,
    ).to(device)
    enc.load_state_dict(state, strict=True)
    enc.eval()
    w = F.softmax(enc.stream_logits, dim=0).detach().cpu().numpy()
    print(f"[encoder] {ENCODER_PATH.name}")
    print(f"  stream weights → pose={w[0]:.3f} L={w[1]:.3f} R={w[2]:.3f} face={w[3]:.3f}")
    return enc


# ── 임베딩 ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_all(
    encoder: LandmarkCrossStreamEncoder,
    segs: dict[str, list[np.ndarray]],
    device: torch.device,
    batch: int = 128,
) -> tuple[np.ndarray, list[str]]:
    vecs, lbls = [], []
    for label, samples in sorted(segs.items()):
        for i in range(0, len(samples), batch):
            chunk = samples[i : i + batch]
            x = torch.from_numpy(np.stack(chunk)).float().to(device)
            e = F.normalize(encoder(x), dim=1).cpu().numpy().astype(np.float32)
            vecs.append(e)
            lbls.extend([label] * len(chunk))
    mat = np.concatenate(vecs, axis=0)
    print(f"[embed] {mat.shape[0]} 벡터, dim={mat.shape[1]}")
    return mat, lbls


# ── FAISS 빌드 ────────────────────────────────────────────────────────────────
def build_faiss(mat: np.ndarray, lbls: list[str], out_dir: Path):
    faiss.normalize_L2(mat)
    idx = faiss.IndexFlatIP(mat.shape[1])
    idx.add(mat)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path  = out_dir / f"{FAISS_NAME}.faiss"
    labels_path = out_dir / f"{FAISS_NAME}.faiss.labels.json"

    faiss.write_index(idx, str(faiss_path))
    labels_path.write_text(json.dumps(lbls, ensure_ascii=False), encoding="utf-8")

    print(f"[faiss] {faiss_path}")
    print(f"  vectors={idx.ntotal}, words={len(set(lbls))}, dim={idx.d}")
    return faiss_path, labels_path


# ── HuggingFace 업로드 ────────────────────────────────────────────────────────
def upload_to_hf(repo_id: str, out_dir: Path):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    print(f"\n[hf] 업로드 대상: {repo_id}")

    create_repo(repo_id, repo_type="model", exist_ok=True, private=False)
    print(f"[hf] 레포지토리 준비 완료")

    faiss_path  = out_dir / f"{FAISS_NAME}.faiss"
    labels_path = out_dir / f"{FAISS_NAME}.faiss.labels.json"
    # HF 경로 = capstone/encoder/, capstone/db/ (기존 repo 구조)
    enc_rel    = HF_ENCODER_PATH
    faiss_rel  = f"{HF_FAISS_PREFIX}/{FAISS_NAME}.faiss"
    labels_rel = f"{HF_FAISS_PREFIX}/{FAISS_NAME}.faiss.labels.json"
    files = [
        (ENCODER_PATH, enc_rel),
        (faiss_path,   faiss_rel),
        (labels_path,  labels_rel),
    ]

    # README.md 생성 (프로젝트 루트에 저장)
    readme = PROJECT_ROOT / "README.md"
    readme.write_text(
        "# KSL Sign Language Encoder (Phase 1)\n\n"
        "## Model\n"
        "- Architecture: `LandmarkCrossStreamEncoder` (4-stream Cross-Attention Transformer)\n"
        "- d_model=256, num_layers=2, nhead=8\n"
        "- Training: AI Hub KSL dataset, NT-Xent loss, Time Warp augmentation\n"
        "- LOO Top-3: **77.6%** on recorded-video (51 words, 152 samples)\n\n"
        "## Files\n"
        f"- `{enc_rel}` — encoder weights\n"
        f"- `{faiss_rel}` — FAISS IndexFlatIP (256-dim, 100 words, jbedu-recordings)\n"
        f"- `{labels_rel}` — FAISS label list\n\n"
        "## Download (project root 기준)\n"
        "```bash\n"
        f"huggingface-cli download {repo_id} \\\n"
        f"  {enc_rel} \\\n"
        f"  {faiss_rel} \\\n"
        f"  {labels_rel} \\\n"
        "  --local-dir .\n"
        "```\n",
        encoding="utf-8",
    )
    files.append((readme, "README.md"))

    for local_path, hf_path in files:
        print(f"  업로드 중: {hf_path} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=hf_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print("✓")

    url = f"https://huggingface.co/{repo_id}"
    print(f"\n[hf] 완료: {url}")
    return url


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo",       required=True,
                   help="HuggingFace repo id (예: username/ksl-encoder)")
    p.add_argument("--out-dir",    default=str(OUT_DIR))
    p.add_argument("--no-upload",  action="store_true",
                   help="FAISS 빌드만 하고 업로드 생략")
    args = p.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"[device] {device}\n")

    out_dir = Path(args.out_dir)

    print("=== 1. 데이터 로딩 ===")
    segs = load_recordings(RECORDINGS_DIR)

    print("\n=== 2. 인코더 로딩 ===")
    encoder = load_encoder(device)

    print("\n=== 3. 임베딩 ===")
    mat, lbls = embed_all(encoder, segs, device)

    print("\n=== 4. FAISS 빌드 ===")
    faiss_path, labels_path = build_faiss(mat, lbls, out_dir)

    if not args.no_upload:
        print("\n=== 5. HuggingFace 업로드 ===")
        upload_to_hf(args.repo, out_dir)


if __name__ == "__main__":
    main()
