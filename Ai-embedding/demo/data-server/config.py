import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL     = os.getenv("DATABASE_URL", "")
DATA_API_KEY     = os.getenv("DATA_API_KEY", "")
JBEDU_VIDEO_ROOT = os.getenv("JBEDU_VIDEO_ROOT", "/data/videos")
RECORDINGS_DIR   = os.getenv("RECORDINGS_DIR", "/data/recordings")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))
