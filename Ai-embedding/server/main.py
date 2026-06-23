import asyncio
import json
import os
from contextlib import asynccontextmanager

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field

import config
from services.embedding import Encoder
from services.preprocessing import convert_word
from services.vectordb import VectorDB


def _hf_download(repo_path: str) -> str:
    print(f"[hf] downloading {repo_path} from {config.HF_REPO_ID}")
    return hf_hub_download(
        repo_id=config.HF_REPO_ID,
        filename=repo_path,
        token=config.HF_TOKEN,
        local_files_only=False,
    )


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    encoder_path = _hf_download(config.HF_ENCODER_PATH)
    print(f"[startup] encoder  : {encoder_path}")
    app.state.encoder = Encoder(encoder_path)
    print(f"[startup]   input_dim={app.state.encoder.input_dim}  "
          f"d_model={app.state.encoder.d_model}  "
          f"num_layers={app.state.encoder.num_layers}")

    faiss_path  = _hf_download(config.HF_FAISS_PATH)
    labels_path = _hf_download(config.HF_LABELS_PATH)
    print(f"[startup] faiss db : {faiss_path}")
    app.state.db = VectorDB(faiss_path, labels_path)
    print(f"[startup]   {app.state.db.num_vectors} vectors / {app.state.db.num_words} words")

    if config.SLLM_ENDPOINT:
        print(f"[startup] sLLM     : remote → {config.SLLM_ENDPOINT}")
    else:
        print("[startup] sLLM     : disabled (SLLM_ENDPOINT not set)")

    print("[startup] ready")
    yield


app = FastAPI(title="KSL Production API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(status: str, content: dict) -> str:
    payload = {"status": status, "content": content}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ── Schema ────────────────────────────────────────────────────────────────────

class WordSegment(BaseModel):
    """
    단어 1개의 키포인트 시퀀스. 각 landmark group은 (T, n_lm * 3) flat 배열.

    landmark 순서 (Node.js POSE_INDICES 기준):
      pose[t]       = [nose(3), L_sh(3), R_sh(3), L_elbow(3), R_elbow(3),
                        L_wrist(3), R_wrist(3), L_hip(3), R_hip(3)]   → 27 floats
      left_hand[t]  = MediaPipe 21 landmarks × 3                       → 63 floats
      right_hand[t] = MediaPipe 21 landmarks × 3                       → 63 floats
      face[t]       = [L_eyebrow×4, R_eyebrow×4, L_eye×3, R_eye×3,
                        nose_tip×1, mouth×4] × 3                       → 57 floats

    T는 단어마다 달라도 됨 (서버가 64프레임으로 보간).
    """
    pose:       list[list[float]] = Field(..., description="T × 27  (9 lm × 3 axes)")
    left_hand:  list[list[float]] = Field(..., description="T × 63  (21 lm × 3 axes)")
    right_hand: list[list[float]] = Field(..., description="T × 63  (21 lm × 3 axes)")
    face:       list[list[float]] = Field(..., description="T × 57  (19 lm × 3 axes)")


class PredictRequest(BaseModel):
    """
    Node.js가 단어 경계를 기준으로 잘라서 보내는 단어 목록.
    배열 순서 = 시간 순서 (index 0이 첫 번째 단어).
    """
    words: list[WordSegment] = Field(..., min_length=1)


# ── /predict ──────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(req: PredictRequest):
    """
    단어 목록을 받아 순서대로 임베딩 → FAISS 검색 후 SSE로 스트리밍 반환.

    SSE event types
    ---------------
    status  — 처리 단계 알림
    word    — 단어 1개 인식 결과
    done    — 전체 완료 (words 배열 + sentence)
    error   — 단어 단위 오류 (해당 idx만, 스트림은 계속)
    """

    async def generate():
        n = len(req.words)
        yield _sse("received", {"n_words": n})

        encoder = app.state.encoder
        db      = app.state.db
        all_candidates: list[list[dict]] = []

        # ── Phase 1: RAG retrieve ─────────────────────────────────────────────
        for i, word_seg in enumerate(req.words):
            try:
                stream = convert_word(word_seg.model_dump(), preset=config.FEATURE_PRESET)
            except Exception as e:
                yield _sse("error", {"idx": i, "message": f"preprocess: {e}"})
                all_candidates.append([])
                continue

            if stream.shape[0] < 2:
                yield _sse("error", {"idx": i, "message": "프레임 부족 (< 2)"})
                all_candidates.append([])
                continue

            emb        = encoder.embed(stream)
            top        = db.search(emb, top_k=min(config.TOP_K, db.num_vectors))
            candidates = [{"word": w, "score": round(s, 4)} for w, s in top]
            all_candidates.append(candidates)
            yield _sse("rag-retrieve", {"idx": i, "total": n})

        # ── Phase 2: sLLM sentence generation ────────────────────────────────
        yield _sse("generate-sentence", {})

        valid_candidates = [c for c in all_candidates if c]

        fallback = " ".join(c[0]["word"] for c in valid_candidates if c)
        sentence = fallback

        if config.SLLM_ENDPOINT and valid_candidates:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(
                        f"{config.SLLM_ENDPOINT}/normalize",
                        json={"candidates": valid_candidates},
                    )
                resp.raise_for_status()
                sentence = resp.json().get("sentence", "") or fallback
            except Exception:
                sentence = fallback

        for i, char in enumerate(sentence):
            yield _sse("sentence-chunk", {"idx": i, "chunk": char})

        yield _sse("done", {"sentence": sentence})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    db = app.state.db
    return {
        "status":      "ok",
        "num_vectors": db.num_vectors,
        "num_words":   db.num_words,
        "encoder":     config.HF_ENCODER_PATH,
        "db":          config.HF_FAISS_PATH,
        "repo":        config.HF_REPO_ID,
        "sllm":        config.SLLM_ENDPOINT or "disabled",
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
