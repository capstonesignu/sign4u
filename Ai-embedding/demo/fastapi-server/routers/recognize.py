from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

router = APIRouter()


class RecognizeRequest(BaseModel):
    keypoints: List[List[float]]
    word_mode: bool = False   # True → 세그멘터 생략, 전체 시퀀스를 단어 1개로
    boundaries: List[int] = []  # 클라이언트 감지 단어 경계 프레임 인덱스


class ResultItem(BaseModel):
    rank: int
    word_id: str
    word: str
    score: float


class RecognizeResponse(BaseModel):
    results: List[ResultItem]


@router.post("/api/recognize", response_model=RecognizeResponse)
async def recognize(req: RecognizeRequest, request: Request):
    """Single-segment recognition: EXP-026 embed → FAISS top-10."""
    app = request.app

    try:
        kp = np.array(req.keypoints, dtype=np.float32)
    except Exception:
        raise HTTPException(400, "Invalid keypoints format")

    if kp.ndim != 2 or kp.shape[1] not in (210, 136):
        raise HTTPException(400, f"Expected (N, 210) or (N, 136), got {kp.shape}")

    if kp.shape[1] == 210:
        from services.preprocessing import convert_stream
        stream = convert_stream(kp, target_preset="B")
    else:
        stream = kp  # already preprocessed by browser

    emb = app.state.exp026_embedding.embed_varlen(stream)
    raw = app.state.vectordb.search(emb, top_k=10)

    items = []
    for i, (word, score) in enumerate(raw):
        items.append(ResultItem(
            rank=i + 1,
            word_id=word,
            word=word,
            score=round(float(score), 4),
        ))

    return RecognizeResponse(results=items)


class CandidateItem(BaseModel):
    word: str
    score: float

class SentenceRecognizeResponse(BaseModel):
    words: List[str]                        # top-1 per segment
    alternatives: List[List[CandidateItem]] # top-k per segment with scores
    spans: List[List[int]]
    n_frames: int


def _velocity_segment(
    stream: np.ndarray,
    threshold: float,
    pause_frames: int,
    min_seg_frames: int = 8,
) -> list[tuple[int, int]]:
    """
    Pause-based segmenter using frame-to-frame L2 velocity on pose+hand dims.
    A word boundary is declared at the midpoint of any pause region
    (consecutive low-velocity frames) that lasts >= pause_frames.
    Returns list of (start, end) inclusive frame index pairs.
    """
    T = stream.shape[0]
    if T < 2:
        return [(0, T - 1)]

    # Pose (14) + left hand (42) + right hand (42) = 98 dims; skip face noise
    dims = stream[:, :98]
    vel  = np.linalg.norm(np.diff(dims, axis=0), axis=1)  # (T-1,)

    # 3-frame smoothing
    vel_smooth = np.convolve(vel, np.ones(3) / 3, mode="same")

    is_pause = vel_smooth < threshold  # (T-1,)

    boundaries: list[int] = []
    pause_start = None
    for i, p in enumerate(is_pause):
        if p:
            if pause_start is None:
                pause_start = i
        else:
            if pause_start is not None:
                length = i - pause_start
                if length >= pause_frames:
                    boundaries.append(pause_start + length // 2)
                pause_start = None
    # trailing pause
    if pause_start is not None:
        length = len(is_pause) - pause_start
        if length >= pause_frames:
            boundaries.append(pause_start + length // 2)

    if not boundaries:
        return [(0, T - 1)]

    cuts = [0] + [b + 1 for b in boundaries] + [T]
    spans = []
    for i in range(len(cuts) - 1):
        sf, ef = cuts[i], cuts[i + 1] - 1
        if ef - sf + 1 >= min_seg_frames:
            spans.append((sf, ef))

    return spans if spans else [(0, T - 1)]


@router.post("/api/recognize/sentence", response_model=SentenceRecognizeResponse)
async def recognize_sentence(req: RecognizeRequest, request: Request):
    """Sentence-level: velocity-pause segmenter → EXP-026 → FAISS top-k."""
    app = request.app

    try:
        kp_raw = np.array(req.keypoints, dtype=np.float32)
    except Exception:
        raise HTTPException(400, "Invalid keypoints format")

    if kp_raw.ndim != 2 or kp_raw.shape[1] not in (210, 136):
        raise HTTPException(400, f"Expected (N, 210) or (N, 136), got {kp_raw.shape}")

    if kp_raw.shape[0] < 10:
        raise HTTPException(400, "Too few frames (minimum 10)")

    if kp_raw.shape[1] == 210:
        from services.preprocessing import convert_stream
        stream = convert_stream(kp_raw, target_preset="B")
    else:
        stream = kp_raw  # already preprocessed by browser
    T = stream.shape[0]
    top_k = app.state.sentence_top_k

    if req.word_mode:
        # 세그멘터 생략 — 전체 시퀀스를 단어 1개로 처리
        emb   = app.state.exp026_embedding.embed_varlen(stream)
        k     = min(top_k, app.state.vectordb.num_vectors)
        raw   = app.state.vectordb.search(emb, top_k=k)
        cands = [CandidateItem(word=label, score=round(float(score), 4))
                 for label, score in raw]
        return SentenceRecognizeResponse(
            words=[cands[0].word] if cands else [],
            alternatives=[cands] if cands else [],
            spans=[[0, T - 1]],
            n_frames=T,
        )

    if req.boundaries:
        # 클라이언트가 감지한 경계 사용 (velocity 스케일 불일치 우회)
        cuts = sorted(set([0] + [b for b in req.boundaries if 0 < b < T] + [T]))
        spans = []
        for i in range(len(cuts) - 1):
            sf, ef = cuts[i], cuts[i + 1] - 1
            if ef - sf + 1 >= 8:
                spans.append((sf, ef))
        if not spans:
            spans = [(0, T - 1)]
    else:
        from config import VELOCITY_THRESHOLD, VELOCITY_PAUSE_FRAMES
        spans = _velocity_segment(
            stream,
            threshold=VELOCITY_THRESHOLD,
            pause_frames=VELOCITY_PAUSE_FRAMES,
        )

    words, alternatives, valid_spans = [], [], []
    for sf, ef in spans:
        sf = max(0, sf); ef = min(T - 1, ef)
        seg = stream[sf:ef + 1]
        if len(seg) < 2:
            continue

        emb = app.state.exp026_embedding.embed_varlen(seg)
        k   = min(top_k, app.state.vectordb.num_vectors)
        raw = app.state.vectordb.search(emb, top_k=k)
        if not raw:
            continue

        cands = [CandidateItem(word=label, score=round(float(score), 4))
                 for label, score in raw]
        words.append(cands[0].word)
        alternatives.append(cands)
        valid_spans.append([int(sf), int(ef)])

    return SentenceRecognizeResponse(
        words=words,
        alternatives=alternatives,
        spans=valid_spans,
        n_frames=T,
    )


@router.get("/api/health")
async def health():
    return {"status": "ok"}


@router.get("/api/config")
async def get_config(request: Request):
    app = request.app
    return {
        "feature_preset": app.state.feature_preset,
        "input_dim": app.state.exp026_embedding.input_dim,
        "d_model": app.state.exp026_embedding.d_model,
        "num_words": app.state.vectordb.num_words,
        "num_vectors": app.state.vectordb.num_vectors,
        "landmarks": {"pose": 9, "left_hand": 21, "right_hand": 21, "face": 19},
    }
