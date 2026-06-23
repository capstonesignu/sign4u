"""
demo-desktop/main.py

Browser captures camera (30fps via getUserMedia) → sends JPEG frames over WebSocket
→ Python runs MediaPipe inference → sends keypoints back over the same WebSocket.

Usage:
    python main.py

Env vars:
    PORT            HTTP port (default 3001)
    FASTAPI_URL     FastAPI inference server (default http://localhost:8001)
    DATA_SERVER_URL data-server URL (default empty → /data/* returns 503)
"""

import asyncio
import json
import os
import queue as thread_queue
import sys
import time
import threading
from pathlib import Path

import urllib.request
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# OpenMP / FAISS 충돌 방지 (macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import aiohttp
from aiohttp import web

# ── Config ────────────────────────────────────────────────────────────────────
PORT            = int(os.environ.get("PORT", 3001))
FASTAPI_URL     = os.environ.get("FASTAPI_URL", "http://localhost:8001")
DATA_SERVER_URL = os.environ.get("DATA_SERVER_URL", "")

BASE_DIR     = Path(__file__).parent
PUBLIC_DIR   = BASE_DIR / "public"
FALLBACK_DIR = BASE_DIR.parent / "demo" / "express-server" / "public"

# ── 인식 모델 경로 (capstone/encoder, capstone/db 구조) ─────────────────────
PROJECT_ROOT    = BASE_DIR.parent
RT_ENCODER_PATH = PROJECT_ROOT / "encoder" / "encoder_cross_stream_best.pt"
RT_FAISS_PATH   = PROJECT_ROOT / "db" / "jbedu_recordings.faiss"
RT_LABELS_PATH  = PROJECT_ROOT / "db" / "jbedu_recordings.faiss.labels.json"

# 프로젝트 모듈 경로 등록
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "demo" / "fastapi-server"))

# ── 인식 모델 전역 상태 ───────────────────────────────────────────────────────
_rt_encoder = None
_rt_faiss   = None
_rt_labels  = []


def _load_rt_models():
    global _rt_encoder, _rt_faiss, _rt_labels
    try:
        import torch
        import faiss
        from model import LandmarkCrossStreamEncoder

        ckpt  = torch.load(str(RT_ENCODER_PATH), map_location="cpu", weights_only=False)
        state = ckpt.get("encoder_state_dict", ckpt)
        enc   = LandmarkCrossStreamEncoder(
            d_model=256, nhead=8, num_layers=2,
            dim_feedforward=512, dropout=0.0, norm_first=True,
        )
        enc.load_state_dict(state, strict=True)
        enc.eval()
        _rt_encoder = enc

        _rt_faiss = faiss.read_index(str(RT_FAISS_PATH))
        with open(str(RT_LABELS_PATH), "r", encoding="utf-8") as f:
            _rt_labels = json.load(f)

        print(f"[rt] 인코더 + FAISS 로드 완료 "
              f"({_rt_faiss.ntotal} vectors, {len(set(_rt_labels))} words)")
    except Exception as e:
        print(f"[rt] 모델 로드 실패 (FASTAPI_URL 프록시로 fallback): {e}")


# ── Confidence 게이트 ─────────────────────────────────────────────────────────
# DB-LOO 측정(diagnose_embedding) 결과 reject 신호 판별력(AUC):
#   절대 score 0.56(무용) < margin12(top1-top2) 0.90 < margin_mean(top1-mean(top2..5)) 0.90
# margin_mean 이 혼동 무리(머리/편두통/두통처럼 top2,3 가 다같이 높은 경우)에 강건하고
# 두 DB 모두 FPR 0% 임계가 존재 → 주 게이트는 margin_mean 사용.
# 임계값(DB별 재보정 필요):
#   현재 배포 prod DB(+flip, 구 인코더): margin_mean θ≈0.11
#   staging DB(flip-free, collapse-fix 인코더) 배포 시: θ≈0.245 로 상향
RT_MARGIN_MEAN_THR = float(os.getenv("RT_MARGIN_MEAN_THR", "0.11"))  # 주 게이트
RT_MARGIN_THR  = float(os.getenv("RT_MARGIN_THR", "0.06"))   # top1-top2 (참고/보조)
RT_SCORE_FLOOR = float(os.getenv("RT_SCORE_FLOOR", "0.50"))  # 쓰레기 입력 컷(느슨)


def _rt_infer(kp_raw: np.ndarray) -> dict:
    import torch
    import torch.nn.functional as F
    import faiss as _faiss
    from services.preprocessing import convert_stream
    from preprocess import temporal_interpolate, TARGET_LENGTH

    if kp_raw.shape[1] == 210:
        stream = convert_stream(kp_raw, target_preset="B")
    else:
        stream = kp_raw

    seg = temporal_interpolate(stream.astype(np.float32), TARGET_LENGTH)

    with torch.no_grad():
        x   = torch.from_numpy(seg).unsqueeze(0)
        emb = F.normalize(_rt_encoder(x)[0], dim=0).cpu().numpy().astype(np.float32)

    q = emb[None, :].copy()
    _faiss.normalize_L2(q)
    fetch_k = min(100, _rt_faiss.ntotal)
    D, I    = _rt_faiss.search(q, fetch_k)

    best: dict = {}
    for dist, idx in zip(D[0], I[0]):
        if 0 <= idx < len(_rt_labels):
            word  = _rt_labels[idx]
            score = float(dist)
            if word not in best or score > best[word]:
                best[word] = score

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:10]

    # margin: top1-top2 (혼동 무리 대비 top1-mean(top2..5)도 함께 제공)
    top1   = ranked[0][1] if ranked else 0.0
    margin = (ranked[0][1] - ranked[1][1]) if len(ranked) >= 2 else None
    tail   = [s for _, s in ranked[1:5]]
    margin_mean = (top1 - sum(tail) / len(tail)) if tail else None

    # reject 게이트: 주 신호 = margin_mean (혼동 무리에 강건, FPR 0% 임계 존재).
    # 단일 후보면 margin_mean=None → score floor 만으로 판정.
    accepted = (
        bool(ranked)
        and top1 >= RT_SCORE_FLOOR
        and (margin_mean is None or margin_mean >= RT_MARGIN_MEAN_THR)
    )

    return {
        "results": [
            {"rank": i + 1, "word": w, "word_id": w, "score": round(s, 4)}
            for i, (w, s) in enumerate(ranked)
        ],
        "top1_score":     round(top1, 4),
        "margin":         round(margin, 4) if margin is not None else None,
        "margin_mean":    round(margin_mean, 4) if margin_mean is not None else None,
        "accepted":       accepted,
        "margin_thr":     RT_MARGIN_MEAN_THR,   # UI 안내용 = 주 게이트 임계
        "margin_thr_t1t2": RT_MARGIN_THR,
    }

MODEL_PATH = BASE_DIR / "holistic_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".ico":  "image/x-icon",
    ".mp4":  "video/mp4",
    ".webm": "video/webm",
}

# ── Landmark serializer ───────────────────────────────────────────────────────
def _lms_to_list(lms, n):
    if not lms:
        return None
    out = []
    for lm in lms[:n]:
        out.append(round(lm.x, 4))
        out.append(round(lm.y, 4))
        out.append(round(lm.z, 4))
    return out


def _ensure_model():
    if not MODEL_PATH.exists():
        print(f"[mediapipe] 모델 다운로드 중... ({MODEL_URL})")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[mediapipe] 모델 저장 완료: {MODEL_PATH}")


# ── MediaPipe 추론 스레드 ─────────────────────────────────────────────────────
# frame_queue: (jpeg_bytes, ws) 튜플
_frame_queue: thread_queue.Queue = thread_queue.Queue(maxsize=2)


def inference_loop():
    _ensure_model()
    options = mp_vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_suppression_threshold=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_suppression_threshold=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
        output_face_blendshapes=False,
        output_segmentation_mask=False,
    )

    start_time = time.monotonic()

    with mp_vision.HolisticLandmarker.create_from_options(options) as landmarker:
        print("[mediapipe] HolisticLandmarker ready")
        while True:
            jpeg_bytes, ws, loop = _frame_queue.get()

            # JPEG 디코딩
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            ts_ms = int((time.monotonic() - start_time) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            try:
                result = landmarker.detect_for_video(mp_image, ts_ms)
            except RuntimeError:
                continue

            payload = json.dumps({
                "pose":      _lms_to_list(result.pose_landmarks, 33),
                "leftHand":  _lms_to_list(result.left_hand_landmarks, 21),
                "rightHand": _lms_to_list(result.right_hand_landmarks, 21),
                "face":      _lms_to_list(result.face_landmarks, 478),
            })

            # 결과를 해당 WebSocket 클라이언트에게 전송
            asyncio.run_coroutine_threadsafe(_safe_send(ws, payload), loop)


async def _safe_send(ws, payload: str):
    try:
        await ws.send_str(payload)
    except Exception:
        pass


# ── WebSocket 핸들러 (양방향) ─────────────────────────────────────────────────
async def ws_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    loop = asyncio.get_running_loop()
    print(f"[ws] client connected")

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                # 브라우저가 보낸 JPEG 프레임 — stale 프레임 드롭 후 큐에 추가
                if _frame_queue.full():
                    try:
                        _frame_queue.get_nowait()
                    except thread_queue.Empty:
                        pass
                _frame_queue.put_nowait((msg.data, ws, loop))

            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                break
    finally:
        print(f"[ws] client disconnected")

    return ws


# ── Static file handler ───────────────────────────────────────────────────────
async def static_handler(request):
    rel = (request.match_info.get("path", "") or "index.html").lstrip("/")
    if not rel:
        rel = "index.html"

    for base in [PUBLIC_DIR, FALLBACK_DIR]:
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            ctype = MIME.get(candidate.suffix.lower(), "application/octet-stream")
            return web.FileResponse(candidate, headers={"Content-Type": ctype})

    return web.Response(status=404, text=f"Not found: {rel}")


# ── Proxy handler ─────────────────────────────────────────────────────────────
def _make_proxy(prefix, base_url):
    async def handler(request):
        if not base_url:
            return web.Response(status=503, text=f"/{prefix}/* service not configured")
        path = request.match_info.get("path", "")
        qs   = ("?" + request.query_string) if request.query_string else ""
        url  = f"{base_url.rstrip('/')}/{prefix}/{path}{qs}"
        body = await request.read()
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    request.method, url,
                    data=body or None,
                    headers=headers,
                    allow_redirects=False,
                ) as resp:
                    content = await resp.read()
                    resp_headers = {
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in ("transfer-encoding", "content-encoding",
                                             "content-length")
                    }
                    return web.Response(body=content, status=resp.status,
                                        headers=resp_headers)
            except aiohttp.ClientConnectorError:
                return web.Response(status=502, text=f"Cannot connect to {base_url}")
    return handler


# ── 인식 API 핸들러 ───────────────────────────────────────────────────────────
async def api_rt_recognize(request):
    if _rt_encoder is None:
        return web.json_response(
            {"error": "인식 모델 미로드 — encoder/encoder_cross_stream_best.pt 확인"},
            status=503,
        )
    try:
        data   = await request.json()
        kp_raw = np.array(data["keypoints"], dtype=np.float32)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)

    if kp_raw.ndim != 2 or kp_raw.shape[1] not in (210, 136):
        return web.json_response(
            {"error": f"Expected (N, 210) or (N, 136), got {list(kp_raw.shape)}"},
            status=400,
        )

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _rt_infer, kp_raw)
    return web.json_response(result)


async def api_rt_config(request):
    if _rt_encoder is not None and _rt_faiss is not None:
        return web.json_response({
            "feature_preset": "B", "input_dim": 136, "d_model": 256,
            "num_words": len(set(_rt_labels)), "num_vectors": _rt_faiss.ntotal,
        })
    return web.json_response({
        "feature_preset": "B", "input_dim": 136,
        "d_model": 256, "num_words": 0, "num_vectors": 0,
    })


# ── App factory ───────────────────────────────────────────────────────────────
def make_app():
    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/api/recognize", api_rt_recognize)
    app.router.add_get("/api/config", api_rt_config)
    app.router.add_route("*", r"/api/{path:.*}", _make_proxy("api", FASTAPI_URL))
    app.router.add_route("*", r"/data/{path:.*}", _make_proxy("data", DATA_SERVER_URL))
    app.router.add_get(r"/{path:.*}", static_handler)
    return app


# ── Entry point ───────────────────────────────────────────────────────────────
async def run():
    _load_rt_models()
    threading.Thread(target=inference_loop, daemon=True).start()

    app    = make_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", PORT)
    await site.start()

    print(f"[server] http://localhost:{PORT}")
    print(f"[server] FastAPI   → {FASTAPI_URL}")
    print(f"[server] Data      → {DATA_SERVER_URL or '(not set)'}")

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(run())
