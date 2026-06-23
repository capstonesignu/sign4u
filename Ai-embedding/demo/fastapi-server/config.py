import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

FEATURE_PRESET = os.getenv("FEATURE_PRESET", "B")

# ── Sentence pipeline (BiLSTM segmenter + EXP-026 encoder) ───────────────────
SEGMENTER_MODEL_PATH = os.getenv(
    "SEGMENTER_MODEL_PATH",
    str(PROJECT_ROOT / "result_seg_h256_l3" / "seg_model.pt"),
)
EXP026_ENCODER_PATH = os.getenv(
    "EXP026_ENCODER_PATH",
    str(PROJECT_ROOT / "result_30fps_fixed_baseline" / "encoder_30fps_fixed_baseline_best.pt"),
)
EXP026_FAISS_PATH = os.getenv(
    "EXP026_FAISS_PATH",
    str(BASE_DIR / "index" / "recorded_30fps.faiss"),
)
EXP026_FAISS_LABELS_PATH = os.getenv(
    "EXP026_FAISS_LABELS_PATH",
    str(BASE_DIR / "index" / "recorded_30fps.faiss.labels.json"),
)
SENTENCE_B_THRESHOLD = float(os.getenv("SENTENCE_B_THRESHOLD", "0.5"))
SENTENCE_TOP_K = int(os.getenv("SENTENCE_TOP_K", "3"))
SENTENCE_FPS = float(os.getenv("SENTENCE_FPS", "15.0"))

# ── Velocity-based segmenter ──────────────────────────────────────────────────
# L2 velocity of pose+hand dims (98-dim) below this → pause frame
VELOCITY_THRESHOLD   = float(os.getenv("VELOCITY_THRESHOLD", "0.05"))
# consecutive pause frames required to be a word boundary (~1.0 s @ 15 fps)
VELOCITY_PAUSE_FRAMES = int(os.getenv("VELOCITY_PAUSE_FRAMES", "8"))
DATASET_PATH = os.getenv(
    "DATASET_PATH",
    str(PROJECT_ROOT / "dataset" / "aihub" / "1.Training"),
)
# 추가 데이터셋 경로 (쉼표 구분). DB prototype에 포함할 검증 농인 데이터 등.
_extra_raw = os.getenv("EXTRA_DATASET_PATHS", "")
EXTRA_DATASET_PATHS: list = [p.strip() for p in _extra_raw.split(",") if p.strip()]
if not EXTRA_DATASET_PATHS:
    EXTRA_DATASET_PATHS = [str(PROJECT_ROOT / "dataset" / "aihub" / "2.Validation")]

WORD_MAPPING_PATH = os.getenv(
    "WORD_MAPPING_PATH",
    str(PROJECT_ROOT / "word_mapping.json"),
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
