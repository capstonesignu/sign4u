"""Local sLLM server — EXAONE-3.5-2.4B + LoRA (transformers+PEFT, MPS).

실행:
    cd local_sLLM
    python -m uvicorn main:app --host 0.0.0.0 --port 8002

    # 별도 터미널에서 ngrok 터널 생성
    ngrok http 8002
    # 출력된 URL을 Railway 환경변수 SLLM_ENDPOINT에 설정
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

ADAPTER_PATH    = os.getenv("ADAPTER_PATH",    str(Path(__file__).parent / "adapter"))
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", str(Path(__file__).parent / "base_model"))
PORT            = int(os.getenv("PORT", "8002"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from sllm_service import FinetunedSLLMService
    app.state.sllm = FinetunedSLLMService(adapter_path=ADAPTER_PATH, base_model_path=BASE_MODEL_PATH)
    yield


app = FastAPI(title="KSL Local sLLM", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class NormalizeRequest(BaseModel):
    candidates: list[list[dict]]


@app.get("/health")
def health():
    return {"status": "ok", "base_model": BASE_MODEL_PATH, "adapter": ADAPTER_PATH}


@app.post("/normalize")
def normalize(req: NormalizeRequest):
    sentence, candidates = app.state.sllm.normalize_with_candidates(req.candidates)
    print("[normalize] 입력:")
    for i, pos in enumerate(req.candidates, 1):
        if pos:
            cands_str = " | ".join(f"{c['word']}({c['score']:.2f})" for c in pos)
            print(f"  위치 {i}: {cands_str}")
    print(f"[normalize] 출력: {sentence!r}")
    return {"sentence": sentence, "candidates": candidates}
