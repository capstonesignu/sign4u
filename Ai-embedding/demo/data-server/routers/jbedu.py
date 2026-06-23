"""
jbedu.py — jbedu 수어 데이터셋 조회/다운로드 API

인증: Authorization: Bearer {DATA_API_KEY}

엔드포인트:
  GET  /data/entries              목록/검색
  GET  /data/entries/{uuid}       단건 메타데이터
  GET  /data/stats                섹션·카테고리별 통계
  GET  /data/keypoints/{uuid}     NPZ 단건 다운로드
  POST /data/keypoints/batch      UUID 목록 → ZIP 다운로드
  GET  /data/keypoints/batch      섹션/카테고리 전체 → ZIP 다운로드
  GET  /data/video/{uuid}         MP4 스트리밍 (Range 지원)
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from typing import List, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter(prefix="/data", tags=["jbedu"])

_security = HTTPBearer()

VIDEO_ROOT: Path = Path(os.getenv("JBEDU_VIDEO_ROOT", "/data/videos"))


# ── 인증 ─────────────────────────────────────────────────────────────────────

def _verify(creds: HTTPAuthorizationCredentials = Depends(_security)) -> None:
    api_key = os.getenv("DATA_API_KEY", "")
    if not api_key or creds.credentials != api_key:
        raise HTTPException(401, "Invalid API key")


# ── DB 헬퍼 ──────────────────────────────────────────────────────────────────

async def _pool(request: Request) -> asyncpg.Pool:
    pool: asyncpg.Pool | None = getattr(request.app.state, "jbedu_pool", None)
    if pool is None:
        raise HTTPException(503, "Database not available")
    return pool


# ── 목록/검색 ────────────────────────────────────────────────────────────────

@router.get("/entries")
async def list_entries(
    request: Request,
    section: Optional[str]  = Query(None, description="단어 / 문장 / 지명 / 회화수어"),
    category: Optional[str] = Query(None),
    keyword: Optional[str]  = Query(None, description="korean_word 부분 일치"),
    has_keypoints: Optional[bool] = Query(None),
    has_video: Optional[bool]     = Query(None),
    limit: int  = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    pool = await _pool(request)
    conditions = []
    args: list = []

    if section:
        args.append(section);      conditions.append(f"section = ${len(args)}")
    if category:
        args.append(category);     conditions.append(f"category = ${len(args)}")
    if keyword:
        args.append(f"%{keyword}%"); conditions.append(f"korean_word ILIKE ${len(args)}")
    if has_keypoints is not None:
        args.append(has_keypoints); conditions.append(f"has_keypoints = ${len(args)}")
    if has_video is not None:
        args.append(has_video);     conditions.append(f"has_video = ${len(args)}")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    args += [limit, offset]

    rows = await pool.fetch(
        f"""
        SELECT uuid, cid, section, category, korean_word, english_word,
               description, video_duration, has_video, has_keypoints, scraped_at
        FROM jbedu_entries
        {where}
        ORDER BY section, category, korean_word
        LIMIT ${len(args)-1} OFFSET ${len(args)}
        """,
        *args,
    )
    total = await pool.fetchval(
        f"SELECT COUNT(*) FROM jbedu_entries {where}", *args[:-2]
    )
    return {"total": total, "limit": limit, "offset": offset, "entries": [dict(r) for r in rows]}


@router.get("/entries/{uuid}")
async def get_entry(
    uuid: str,
    request: Request,
    _: None = Depends(_verify),
):
    pool = await _pool(request)
    row = await pool.fetchrow(
        """
        SELECT uuid, cid, section, category, korean_word, english_word,
               description, video_duration, video_path, has_video, has_keypoints,
               scraped_at, uploaded_at
        FROM jbedu_entries WHERE uuid = $1
        """,
        uuid,
    )
    if row is None:
        raise HTTPException(404, "Entry not found")
    return dict(row)


@router.get("/stats")
async def stats(request: Request):
    pool = await _pool(request)
    rows = await pool.fetch(
        "SELECT section, category, total, with_video, with_keypoints FROM jbedu_stats"
    )
    return [dict(r) for r in rows]


# ── 키포인트 다운로드 ─────────────────────────────────────────────────────────

@router.get("/keypoints/{uuid}")
async def download_keypoints(
    uuid: str,
    request: Request,
    _: None = Depends(_verify),
):
    """단건 NPZ 다운로드."""
    pool = await _pool(request)
    row = await pool.fetchrow(
        "SELECT korean_word, keypoints FROM jbedu_entries WHERE uuid = $1", uuid
    )
    if row is None:
        raise HTTPException(404, "Entry not found")
    if not row["keypoints"]:
        raise HTTPException(404, "Keypoints not available for this entry")

    safe_name = row["korean_word"].replace("/", "_") or uuid
    return Response(
        content=bytes(row["keypoints"]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_{uuid[:8]}.npz"'},
    )


async def _build_zip(rows) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for row in rows:
            if not row["keypoints"]:
                continue
            safe = row["korean_word"].replace("/", "_") or row["uuid"]
            fname = f"{row['section']}/{row['category']}/{safe}_{row['uuid'][:8]}.npz"
            zf.writestr(fname, bytes(row["keypoints"]))
    buf.seek(0)
    return buf.read()


@router.post("/keypoints/batch")
async def batch_keypoints_by_uuids(
    request: Request,
    body: dict,
    _: None = Depends(_verify),
):
    """UUID 목록 → ZIP."""
    uuids: List[str] = body.get("uuids", [])
    if not uuids:
        raise HTTPException(400, "uuids list is required")
    if len(uuids) > 2000:
        raise HTTPException(400, "Maximum 2000 UUIDs per request")

    pool = await _pool(request)
    rows = await pool.fetch(
        "SELECT uuid, section, category, korean_word, keypoints FROM jbedu_entries WHERE uuid = ANY($1::text[])",
        uuids,
    )
    zip_bytes = await _build_zip(rows)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="keypoints_batch.zip"'},
    )


@router.get("/keypoints/batch")
async def batch_keypoints_by_filter(
    request: Request,
    section: Optional[str]  = Query(None),
    category: Optional[str] = Query(None),
    _: None = Depends(_verify),
):
    """섹션/카테고리 전체 → ZIP (keypoints 있는 것만)."""
    pool = await _pool(request)
    conditions = ["has_keypoints = TRUE"]
    args: list = []
    if section:
        args.append(section);  conditions.append(f"section = ${len(args)}")
    if category:
        args.append(category); conditions.append(f"category = ${len(args)}")

    where = "WHERE " + " AND ".join(conditions)
    rows = await pool.fetch(
        f"SELECT uuid, section, category, korean_word, keypoints FROM jbedu_entries {where}",
        *args,
    )
    if not rows:
        raise HTTPException(404, "No keypoints found for the given filter")

    zip_bytes = await _build_zip(rows)
    label = f"{section or 'all'}_{category or 'all'}"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="keypoints_{label}.zip"'},
    )


# ── 영상 업로드 (관리자용) ───────────────────────────────────────────────────

@router.post("/admin/upload/{uuid}")
async def upload_video_file(
    uuid: str,
    request: Request,
    _: None = Depends(_verify),
):
    """비디오 파일을 Volume에 저장하고 DB 업데이트. 바디: raw MP4 바이너리."""
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    video_file = VIDEO_ROOT / f"{uuid}.mp4"

    try:
        with open(video_file, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
    except Exception as e:
        video_file.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to write video: {e}")

    size = video_file.stat().st_size
    pool = await _pool(request)
    await pool.execute(
        "UPDATE jbedu_entries SET has_video = TRUE, video_path = $1 WHERE uuid = $2",
        f"{uuid}.mp4",
        uuid,
    )
    return {"uuid": uuid, "size_bytes": size}


# ── 영상 스트리밍 ─────────────────────────────────────────────────────────────

@router.get("/video/{uuid}")
async def stream_video(
    uuid: str,
    request: Request,
):
    """MP4 스트리밍 (HTTP Range 지원)."""
    pool = await _pool(request)
    row = await pool.fetchrow(
        "SELECT video_path, has_video FROM jbedu_entries WHERE uuid = $1", uuid
    )
    if row is None:
        raise HTTPException(404, "Entry not found")
    if not row["has_video"] or not row["video_path"]:
        raise HTTPException(404, "Video not available for this entry")

    video_file = VIDEO_ROOT / row["video_path"]
    if not video_file.exists():
        raise HTTPException(404, "Video file not found on server")

    file_size = video_file.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        # Range: bytes=start-end
        range_val = range_header.replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0])
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def _iter():
            with open(video_file, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            _iter(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )

    def _iter_full():
        with open(video_file, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        _iter_full(),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )
