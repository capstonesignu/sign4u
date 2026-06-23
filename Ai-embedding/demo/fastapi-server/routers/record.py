"""
record.py

키포인트 녹화 저장 엔드포인트.
프론트에서 받은 raw keypoints (T, 210) → NPZ (pose/left_hand/right_hand/face) 변환 저장.
"""
import io
import re
import time
from pathlib import Path
from typing import List
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

RECORDINGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "dataset" / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)

_POSE_START, _POSE_END  = 0,  9
_LH_START,   _LH_END    = 9,  30
_RH_START,   _RH_END    = 30, 51
_FACE_START, _FACE_END   = 51, 70


class RecordSaveRequest(BaseModel):
    keypoints: List[List[float]]   # (T, 210) raw MediaPipe landmarks
    label: str
    fps: float = 15.0


@router.post("/api/record/save")
async def record_save(req: RecordSaveRequest):
    """Raw keypoints → NPZ 변환 후 다운로드."""
    if not req.keypoints:
        raise HTTPException(400, "keypoints가 비어 있습니다")

    kp = np.array(req.keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] != 210:
        raise HTTPException(400, f"Expected (T, 210), got {kp.shape}")

    T = kp.shape[0]
    frames = kp.reshape(T, 70, 3)

    pose       = frames[:, _POSE_START:_POSE_END,  :]
    left_hand  = frames[:, _LH_START  :_LH_END,   :]
    right_hand = frames[:, _RH_START  :_RH_END,   :]
    face       = frames[:, _FACE_START:_FACE_END,  :]

    safe_label = re.sub(r"[^\w가-힣]", "_", req.label.strip()) or "unknown"
    timestamp  = int(time.time())
    filename   = f"{safe_label}_{timestamp}.npz"
    save_path  = RECORDINGS_DIR / filename

    np.savez(str(save_path), fps=np.float32(req.fps),
             pose=pose, left_hand=left_hand, right_hand=right_hand, face=face)

    buf = io.BytesIO()
    np.savez(buf, fps=np.float32(req.fps),
             pose=pose, left_hand=left_hand, right_hand=right_hand, face=face)
    buf.seek(0)

    encoded_name = quote(filename, safe="")
    return StreamingResponse(
        buf,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_name}",
            "X-Filename": encoded_name,
            "X-Frames": str(T),
            "Access-Control-Expose-Headers": "X-Filename, X-Frames, Content-Disposition",
        },
    )


@router.get("/api/record/list")
async def record_list():
    files = sorted(RECORDINGS_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {"filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)}
        for f in files[:50]
    ]
