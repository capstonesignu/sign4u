/**
 * MediaPipe HolisticLandmarker keypoint extraction (Tasks API).
 *
 * Tasks API로 전환 — aihub-keypoint/mediapipe_config.yaml과 동일한 파라미터.
 * (기존: @mediapipe/holistic@0.5 Solutions API)
 *
 * 70 landmarks × 3 axes = 210 dim/frame:
 *   Pose 9 (from 33): nose, shoulders, elbows, wrists, hips
 *   Left Hand 21 (all)
 *   Right Hand 21 (all)
 *   Face 19 (from 478): eyebrows, eyes, mouth, nose tip
 */

const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24];

const FACE_INDICES = [
  46, 53, 52, 65,     // left eyebrow
  276, 283, 282, 295, // right eyebrow
  33, 159, 133,       // left eye
  362, 386, 263,      // right eye
  1,                  // nose tip
  61, 291, 0, 17,     // mouth
];

// pose draw connections (원본 33개 index 기준, adapter 후 33개 유지)
const POSE_DRAW_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24],
];

let holisticLandmarker = null;
let animFrameId = null;
let onResultsCallback = null;
let isHolisticReady = false;

// Tasks API 결과 → 기존 {poseLandmarks, leftHandLandmarks, ...} 형태로 변환
function _adaptResult(result) {
  return {
    poseLandmarks:      result.poseLandmarks.length      > 0 ? result.poseLandmarks[0]      : null,
    leftHandLandmarks:  result.leftHandLandmarks.length  > 0 ? result.leftHandLandmarks[0]  : null,
    rightHandLandmarks: result.rightHandLandmarks.length > 0 ? result.rightHandLandmarks[0] : null,
    faceLandmarks:      result.faceLandmarks.length      > 0 ? result.faceLandmarks[0]      : null,
  };
}

function extractKeypoints(results) {
  const inBounds = (lm) => lm.x >= 0 && lm.x <= 1 && lm.y >= 0 && lm.y <= 1;
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

// nullable 구조체 → flat (T, 210) 배열로 변환 (서버 전송용)
function flattenKeypoints(kp) {
  return [
    ...(kp.pose      ?? new Array(9  * 3).fill(0)),
    ...(kp.leftHand  ?? new Array(21 * 3).fill(0)),
    ...(kp.rightHand ?? new Array(21 * 3).fill(0)),
    ...(kp.face      ?? new Array(19 * 3).fill(0)),
  ];
}

// videoW×videoH: 카메라 원본 해상도, displayW×displayH: 캔버스 CSS 표시 크기
// object-fit:cover 와 동일한 크롭 변환 적용
function drawKeypoints(canvasCtx, results, displayW, displayH, videoW, videoH) {
  canvasCtx.clearRect(0, 0, displayW, displayH);

  // object-fit: cover 보정: scale = max(dW/vW, dH/vH), 초과분은 양쪽 균등 크롭
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

async function initMediaPipe(videoEl, canvasEl) {
  const canvasCtx = canvasEl.getContext("2d");

  const { HolisticLandmarker, FilesetResolver } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/+esm"
  );

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm"
  );

  holisticLandmarker = await HolisticLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
      delegate: "CPU",
    },
    runningMode: "VIDEO",
    minFaceDetectionConfidence: 0.5,
    minFaceSuppressionThreshold: 0.5,
    minFaceLandmarksConfidence: 0.5,
    minPoseDetectionConfidence: 0.5,
    minPoseSuppressionThreshold: 0.5,
    minPoseLandmarksConfidence: 0.5,
    minHandLandmarksConfidence: 0.5,
    outputFaceBlendshapes: false,
    outputSegmentationMasks: false,
  });

  // 웹캠 스트림 직접 열기 (camera_utils 대체)
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
  });
  videoEl.srcObject = stream;
  await new Promise((resolve) => { videoEl.onloadedmetadata = resolve; });
  videoEl.play();

  function detectLoop(now) {
    animFrameId = requestAnimationFrame(detectLoop);
    if (videoEl.readyState < 2) return;

    // 캔버스 크기 = CSS 표시 크기 (object-fit:cover 보정의 기준)
    const displayW = canvasEl.offsetWidth  || 480;
    const displayH = canvasEl.offsetHeight || 360;
    if (canvasEl.width  !== displayW) canvasEl.width  = displayW;
    if (canvasEl.height !== displayH) canvasEl.height = displayH;

    const videoW = videoEl.videoWidth  || displayW;
    const videoH = videoEl.videoHeight || displayH;

    const result  = holisticLandmarker.detectForVideo(videoEl, now);
    const adapted = _adaptResult(result);

    drawKeypoints(canvasCtx, adapted, displayW, displayH, videoW, videoH);
    if (onResultsCallback) onResultsCallback(adapted);
  }

  isHolisticReady = true;
  detectLoop();
  return true;
}

function setOnResults(callback) {
  onResultsCallback = callback;
}
