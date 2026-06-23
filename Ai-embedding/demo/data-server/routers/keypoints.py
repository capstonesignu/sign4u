"""
keypoints.py — 키포인트 NPZ 저장 엔드포인트

POST /data/keypoints/save
  body: {keypoints: List[List[float]] (T×210), fps: float, label: str}
  → 단어별 디렉터리에 NPZ 저장, {ok, saved_npz, frames} 반환

GET  /data/keypoints/list?label=...
  → 저장된 파일 목록 반환
"""
from __future__ import annotations

import re
import time
import urllib.parse
from pathlib import Path
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config

router = APIRouter(prefix="/data/keypoints", tags=["keypoints"])

_RECORDINGS_DIR = Path(config.RECORDINGS_DIR)
_UUID_RE = re.compile(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
)


def _uuid_from_stem(stem: str) -> str | None:
    """파일명 stem에서 UUID 접두사 추출. UUID로 시작하지 않으면 None."""
    part = stem.split("_", 1)[0]
    return part if _UUID_RE.match(part) else None


class SaveRequest(BaseModel):
    keypoints: List[List[float]]   # (T, 210) flat MediaPipe landmarks
    fps: float = 30.0
    label: str
    recorder_name: str = ""
    uuid: str = ""                 # jbedu entry UUID (있으면 파일명 접두사로 사용)


@router.post("/save")
async def save_keypoints(req: SaveRequest):
    label = req.label.strip()
    if not label:
        raise HTTPException(400, "label이 비어 있습니다")

    kp = np.array(req.keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] != 210:
        raise HTTPException(400, f"Expected (T, 210), got shape {list(kp.shape)}")
    if kp.shape[0] < 8:
        raise HTTPException(400, "프레임이 너무 적습니다 (최소 8)")

    T = kp.shape[0]
    pose       = kp[:, 0:27].reshape(T, 9, 3)
    left_hand  = kp[:, 27:90].reshape(T, 21, 3)
    right_hand = kp[:, 90:153].reshape(T, 21, 3)
    face       = kp[:, 153:210].reshape(T, 19, 3)

    out_dir = _RECORDINGS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    ts         = int(time.time() * 1000)
    uuid_clean = req.uuid.strip()
    filename   = f"{uuid_clean}_{ts}.npz" if _UUID_RE.match(uuid_clean) else f"{label}_{ts}.npz"
    out_path = out_dir / filename
    duration = float(T - 1) / req.fps if req.fps > 0 else 0.0

    np.savez_compressed(
        str(out_path),
        pose=pose,
        left_hand=left_hand,
        right_hand=right_hand,
        face=face,
        fps=np.float32(req.fps),
        duration=np.float32(duration),
        recorder=np.array(req.recorder_name.strip()),
    )

    return {"ok": True, "saved_npz": str(out_path), "frames": T, "label": label, "recorder": req.recorder_name.strip()}


@router.get("/list")
async def list_keypoints(label: str = ""):
    if not _RECORDINGS_DIR.exists():
        return {"files": []}
    if label:
        word_dir = _RECORDINGS_DIR / label.strip()
        files = sorted(word_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True) \
                if word_dir.exists() else []
        return {"label": label, "files": [f.name for f in files]}

    words = {}
    uuids = {}
    for word_dir in sorted(_RECORDINGS_DIR.iterdir()):
        if not word_dir.is_dir():
            continue
        files = list(word_dir.glob("*.npz"))
        if not files:
            continue
        words[word_dir.name] = len(files)
        for f in files:
            uid = _uuid_from_stem(f.stem)
            if uid:
                uuids[uid] = uuids.get(uid, 0) + 1
    return {"words": words, "uuids": uuids}


@router.delete("/{label}/{filename}")
async def delete_keypoint_file(label: str, filename: str):
    if not filename.endswith(".npz"):
        raise HTTPException(400, "npz 파일만 삭제 가능합니다")
    path = _RECORDINGS_DIR / label / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, f"{label}/{filename} 을 찾을 수 없습니다")
    path.unlink()
    return {"ok": True, "deleted": filename}


@router.delete("/{label}")
async def delete_keypoint_label(label: str):
    import shutil
    word_dir = _RECORDINGS_DIR / label
    if not word_dir.exists():
        raise HTTPException(404, f"'{label}' 키포인트가 없습니다")
    count = len(list(word_dir.glob("*.npz")))
    shutil.rmtree(word_dir)
    return {"ok": True, "label": label, "deleted": count}


@router.get("/download/{label}/{filename}")
async def download_keypoints(label: str, filename: str):
    if not filename.endswith(".npz"):
        raise HTTPException(400, "npz 파일만 다운로드 가능합니다")
    path = _RECORDINGS_DIR / label / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(404, f"{label}/{filename} 을 찾을 수 없습니다")
    encoded = urllib.parse.quote(filename)
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"},
    )


@router.get("/download/{label}")
async def download_label_zip(label: str):
    """특정 단어의 모든 npz를 zip으로 묶어 반환"""
    import io, zipfile
    word_dir = _RECORDINGS_DIR / label
    if not word_dir.exists():
        raise HTTPException(404, f"'{label}' 키포인트가 없습니다")
    files = sorted(word_dir.glob("*.npz"))
    if not files:
        raise HTTPException(404, f"'{label}' 저장된 파일이 없습니다")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, f.name)
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    encoded = urllib.parse.quote(label)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}.zip"},
    )


@router.get("/download")
async def download_all_zip():
    """전체 recordings 폴더를 zip으로 반환"""
    import io, zipfile
    if not _RECORDINGS_DIR.exists():
        raise HTTPException(404, "저장된 키포인트가 없습니다")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(_RECORDINGS_DIR.rglob("*.npz")):
            zf.write(f, f.relative_to(_RECORDINGS_DIR))
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=recordings.zip"},
    )
