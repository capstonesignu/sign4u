import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN    = os.getenv("HF_TOKEN")
HF_REPO_ID  = os.getenv("HF_REPO_ID")

ENCODER_FILENAME      = os.getenv("ENCODER_FILENAME")
FAISS_FILENAME        = os.getenv("FAISS_FILENAME")
FAISS_LABELS_FILENAME = os.getenv("FAISS_LABELS_FILENAME")

HF_ENCODER_PATH = f"capstone/encoder/{ENCODER_FILENAME}"
HF_FAISS_PATH   = f"capstone/db/{FAISS_FILENAME}"
HF_LABELS_PATH  = f"capstone/db/{FAISS_LABELS_FILENAME}"

# ── Feature preset ────────────────────────────────────────────────────────────
FEATURE_PRESET = os.getenv("FEATURE_PRESET")

# ── FAISS search ──────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K"))

# ── sLLM endpoint (로컬 Mac, ngrok으로 공개) ──────────────────────────────────
SLLM_ENDPOINT = (os.getenv("SLLM_ENDPOINT") or "").rstrip("/")

# ── Server ────────────────────────────────────────────────────────────────────
HOST = os.getenv("HOST")
PORT = int(os.getenv("PORT"))
