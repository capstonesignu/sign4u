import os
from contextlib import asynccontextmanager

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_MPS_DISABLE"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import config
from services.embedding import EmbeddingServiceExp026
from services.vectordb import VectorDBService
from services.segmenter import SegmenterService


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[startup] Loading EXP-026 encoder from {config.EXP026_ENCODER_PATH}")
    app.state.exp026_embedding = EmbeddingServiceExp026(config.EXP026_ENCODER_PATH)

    print(f"[startup] Loading EXP-026 FAISS from {config.EXP026_FAISS_PATH}")
    app.state.vectordb = VectorDBService(
        config.EXP026_FAISS_PATH,
        labels_path=config.EXP026_FAISS_LABELS_PATH,
    )
    print(
        f"[startup] FAISS: {app.state.vectordb.num_vectors} vectors, "
        f"{app.state.vectordb.num_words} words"
    )

    print(f"[startup] Loading segmenter from {config.SEGMENTER_MODEL_PATH}")
    try:
        app.state.segmenter = SegmenterService(config.SEGMENTER_MODEL_PATH)
    except FileNotFoundError:
        print("[startup] Segmenter model not found — skipping (velocity-based segmenter will be used)")
        app.state.segmenter = None

    app.state.sentence_b_threshold = config.SENTENCE_B_THRESHOLD
    app.state.sentence_top_k = config.SENTENCE_TOP_K
    app.state.feature_preset = config.FEATURE_PRESET
    print("[startup] Ready")
    yield


app = FastAPI(title="KSL Sign Language Recognition API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from routers.recognize import router
from routers.record import router as record_router
from routers.db import router as db_router
app.include_router(router)
app.include_router(record_router)
app.include_router(db_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
