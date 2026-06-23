/**
 * ws_camera.js — demo-desktop용 MediaPipe 대체 모듈
 *
 * 브라우저 getUserMedia(30fps) → JPEG → WebSocket → Python MediaPipe → keypoints JSON
 *
 * 공개 API (mediapipe.js와 동일):
 *   initMediaPipe(videoEl, canvasEl)
 *   setOnResults(callback)
 *   extractKeypoints(results)
 *   flattenKeypoints(kp)
 *   drawKeypoints(ctx, results, dW, dH, vW, vH)
 */

// ── 랜드마크 인덱스 상수 ──────────────────────────────────────────────────────
const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24];

const FACE_INDICES = [
  46, 53, 52, 65,
  276, 283, 282, 295,
  33, 159, 133,
  362, 386, 263,
  1,
  61, 291, 0, 17,
];

const POSE_DRAW_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
];

// ── 상태 ──────────────────────────────────────────────────────────────────────
let _primaryCallback   = null;   // setOnResults — 메인 앱 콜백
let _extraCallbacks    = [];     // addOnResults — 추가 탭 콜백 (setOnResults로 삭제 안됨)
let _ws                = null;

// ── flat float 배열 → [{x,y,z}] 변환 ─────────────────────────────────────────
function _flatToLms(arr, n) {
  if (!arr) return null;
  const out = [];
  for (let i = 0; i < n; i++) {
    out.push({ x: arr[i * 3], y: arr[i * 3 + 1], z: arr[i * 3 + 2] });
  }
  return out;
}

// ── extractKeypoints ──────────────────────────────────────────────────────────
function extractKeypoints(results) {
  const inBounds   = (lm) => lm.x >= 0 && lm.x <= 1 && lm.y >= 0 && lm.y <= 1;
  const extractLms = (lms, indices) => {
    if (!lms) return null;
    const arr = [];
    for (const idx of (indices ?? lms.map((_, i) => i))) {
      const lm = lms[idx];
      inBounds(lm) ? arr.push(lm.x, lm.y, lm.z) : arr.push(0, 0, 0);
    }
    return arr;
  };
  return {
    pose:      extractLms(results.poseLandmarks, POSE_INDICES),
    leftHand:  extractLms(results.leftHandLandmarks),
    rightHand: extractLms(results.rightHandLandmarks),
    face:      extractLms(results.faceLandmarks, FACE_INDICES),
  };
}

// ── flattenKeypoints ──────────────────────────────────────────────────────────
function flattenKeypoints(kp) {
  return [
    ...(kp.pose      ?? new Array(9  * 3).fill(0)),
    ...(kp.leftHand  ?? new Array(21 * 3).fill(0)),
    ...(kp.rightHand ?? new Array(21 * 3).fill(0)),
    ...(kp.face      ?? new Array(19 * 3).fill(0)),
  ];
}

// ── drawKeypoints ─────────────────────────────────────────────────────────────
function drawKeypoints(canvasCtx, results, displayW, displayH, videoW, videoH) {
  canvasCtx.clearRect(0, 0, displayW, displayH);

  const scale = Math.max(displayW / videoW, displayH / videoH);
  const offX  = (videoW * scale - displayW) / 2;
  const offY  = (videoH * scale - displayH) / 2;
  const toX   = (nx) => nx * videoW * scale - offX;
  const toY   = (ny) => ny * videoH * scale - offY;

  if (results.poseLandmarks) {
    canvasCtx.strokeStyle = "#00FF00";
    canvasCtx.lineWidth = 2;
    for (const [i, j] of POSE_DRAW_CONNECTIONS) {
      const a = results.poseLandmarks[i];
      const b = results.poseLandmarks[j];
      if (!a || !b) continue;
      canvasCtx.beginPath();
      canvasCtx.moveTo(toX(a.x), toY(a.y));
      canvasCtx.lineTo(toX(b.x), toY(b.y));
      canvasCtx.stroke();
    }
    canvasCtx.fillStyle = "#00FF00";
    for (const idx of POSE_INDICES) {
      const lm = results.poseLandmarks[idx];
      if (!lm) continue;
      canvasCtx.beginPath();
      canvasCtx.arc(toX(lm.x), toY(lm.y), 4, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  }

  if (results.leftHandLandmarks) {
    canvasCtx.fillStyle = "#FF0000";
    for (const lm of results.leftHandLandmarks) {
      canvasCtx.beginPath();
      canvasCtx.arc(toX(lm.x), toY(lm.y), 2, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  }

  if (results.rightHandLandmarks) {
    canvasCtx.fillStyle = "#0000FF";
    for (const lm of results.rightHandLandmarks) {
      canvasCtx.beginPath();
      canvasCtx.arc(toX(lm.x), toY(lm.y), 2, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  }

  if (results.faceLandmarks) {
    canvasCtx.fillStyle = "#FFFF00";
    for (const idx of FACE_INDICES) {
      const lm = results.faceLandmarks[idx];
      if (!lm) continue;
      canvasCtx.beginPath();
      canvasCtx.arc(toX(lm.x), toY(lm.y), 2, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  }
}

// ── setOnResults / addOnResults ──────────────────────────────────────────────
function setOnResults(callback) {
  _primaryCallback = callback;
}

function addOnResults(callback) {
  _extraCallbacks.push(callback);
}

// ── initMediaPipe ─────────────────────────────────────────────────────────────
async function initMediaPipe(videoEl, canvasEl) {
  const canvasCtx = canvasEl.getContext("2d");

  // 1. 카메라 스트림 열기 (화면 표시 + db_record MediaRecorder 용)
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, frameRate: { ideal: 30 } },
  });
  videoEl.srcObject = stream;
  await new Promise((resolve) => { videoEl.onloadedmetadata = resolve; });
  videoEl.play();

  // 2. 프레임 캡처용 오프스크린 캔버스 (320×240 — MediaPipe 처리 속도 최적화)
  const offscreen = document.createElement("canvas");
  offscreen.width  = 320;
  offscreen.height = 240;
  const offCtx = offscreen.getContext("2d");

  // 3. WebSocket 연결
  const wsUrl = `ws://${location.host}/ws`;
  _ws = new WebSocket(wsUrl);
  _ws.binaryType = "arraybuffer";

  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("WebSocket 연결 타임아웃")), 5000);
    _ws.onopen  = () => { clearTimeout(timer); resolve(); };
    _ws.onerror = () => { clearTimeout(timer); reject(new Error("WebSocket 연결 실패: " + wsUrl)); };
  });

  // 4. JPEG 프레임을 30fps로 Python에 전송
  let _sending = false;
  setInterval(() => {
    if (_ws.readyState !== WebSocket.OPEN || _sending) return;
    _sending = true;
    offCtx.drawImage(videoEl, 0, 0, 320, 240);
    offscreen.toBlob((blob) => {
      if (blob && _ws.readyState === WebSocket.OPEN) {
        _ws.send(blob);
      }
      _sending = false;
    }, "image/jpeg", 0.85);
  }, 1000 / 30);

  // 5. 키포인트 수신 (text JSON) → 그리기 + 콜백
  _ws.onmessage = (event) => {
    if (typeof event.data !== "string") return;
    const data = JSON.parse(event.data);

    const results = {
      poseLandmarks:      _flatToLms(data.pose,      33),
      leftHandLandmarks:  _flatToLms(data.leftHand,  21),
      rightHandLandmarks: _flatToLms(data.rightHand, 21),
      faceLandmarks:      _flatToLms(data.face,      478),
    };

    const displayW = canvasEl.offsetWidth  || 480;
    const displayH = canvasEl.offsetHeight || 360;
    if (canvasEl.width  !== displayW) canvasEl.width  = displayW;
    if (canvasEl.height !== displayH) canvasEl.height = displayH;

    const videoW = videoEl.videoWidth  || displayW;
    const videoH = videoEl.videoHeight || displayH;

    drawKeypoints(canvasCtx, results, displayW, displayH, videoW, videoH);
    if (_primaryCallback) _primaryCallback(results);
    for (const cb of _extraCallbacks) cb(results);
  };

  _ws.onclose = () => console.warn("[ws_camera] WebSocket 연결 종료");

  return true;
}
