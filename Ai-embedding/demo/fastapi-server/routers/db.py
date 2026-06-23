"""
db.py — DB 단어 추가 엔드포인트.
raw keypoints (T, 210) → convert_stream(preset B) → embed_varlen → vectordb.add → save.
Also saves raw keypoints as .npz to dataset/recorded-video/{label}/.
"""
import time
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

import config

router = APIRouter()

RECORDED_VIDEO_DIR = Path(config.PROJECT_ROOT) / "dataset" / "recorded-video"


def _save_npz(kp: np.ndarray, label: str, fps: float) -> Path:
    """(T, 210) raw → split into parts → save as npz."""
    T = kp.shape[0]
    pose       = kp[:, 0:27].reshape(T, 9, 3)
    left_hand  = kp[:, 27:90].reshape(T, 21, 3)
    right_hand = kp[:, 90:153].reshape(T, 21, 3)
    face       = kp[:, 153:210].reshape(T, 19, 3)

    out_dir = RECORDED_VIDEO_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = int(time.time() * 1000)
    out_path = out_dir / f"{label}_{ts}.npz"
    duration = float(T - 1) / fps if fps > 0 else 0.0
    np.savez_compressed(
        out_path,
        pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
        face=face,
        fps=np.float32(fps),
        duration=np.float32(duration),
    )
    return out_path


class DBAddRequest(BaseModel):
    keypoints: List[List[float]]  # (T, 210) raw MediaPipe
    fps: float = 30.0
    label: str


@router.post("/api/db/add")
async def db_add(req: DBAddRequest, request: Request):
    """단어 1개 녹화 → FAISS DB에 추가 + npz 저장."""
    app = request.app

    kp = np.array(req.keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] != 210:
        raise HTTPException(400, f"Expected (T, 210), got {kp.shape}")
    if kp.shape[0] < 8:
        raise HTTPException(400, "Too few frames (minimum 8)")

    label = req.label.strip()
    if not label:
        raise HTTPException(400, "label이 비어 있습니다")

    npz_path = _save_npz(kp, label, fps=req.fps)

    return {
        "success": True,
        "word": label,
        "saved_npz": str(npz_path),
    }


@router.get("/api/db/info")
async def db_info(request: Request):
    """현재 DB 통계 반환."""
    vdb = request.app.state.vectordb
    recent = list(dict.fromkeys(reversed(vdb.labels)))[:15]
    return {
        "num_vectors": vdb.num_vectors,
        "num_words": vdb.num_words,
        "recent_words": recent,
    }


@router.get("/api/db/words")
async def db_words(request: Request):
    """FAISS 단어별 벡터 수 + recorded-video 디렉토리 녹화 파일 수."""
    vdb = request.app.state.vectordb

    faiss_counts: dict = {}
    for lbl in vdb.labels:
        faiss_counts[lbl] = faiss_counts.get(lbl, 0) + 1

    rec_counts: dict = {}
    if RECORDED_VIDEO_DIR.exists():
        for word_dir in RECORDED_VIDEO_DIR.iterdir():
            if word_dir.is_dir():
                n = len(list(word_dir.glob("*.npz")))
                if n > 0:
                    rec_counts[word_dir.name] = n

    all_words = sorted(set(list(faiss_counts) + list(rec_counts)))
    words = [
        {"word": w, "vectors": faiss_counts.get(w, 0), "recorded": rec_counts.get(w, 0)}
        for w in all_words
    ]
    return {"words": words}
