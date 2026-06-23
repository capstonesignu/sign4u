/**
 * useMediaPipe — MediaPipe HolisticLandmarker 초기화 및 감지 루프
 *
 * @mediapipe/tasks-vision@0.10.15 (Tasks API, CDN dynamic import)
 * 모델: holistic_landmarker/float16/latest
 * Delegate: CPU
 *
 * 사용법:
 *   const { isReady, error, results } = useMediaPipe(videoRef, canvasRef);
 */

import { useEffect, useRef, useState, useCallback } from "react";
import {
  adaptResult,
  drawKeypoints,
  POSE_INDICES,
  FACE_INDICES,
  POSE_DRAW_CONNECTIONS,
} from "../utils/keypoints";

// 캔버스에 랜드마크 그리기
function drawOnCanvas(canvasCtx, results, width, height) {
  canvasCtx.clearRect(0, 0, width, height);

  // Pose skeleton
  if (results.poseLandmarks) {
    canvasCtx.strokeStyle = "#00FF00";
    canvasCtx.lineWidth = 2;
    for (const [i, j] of POSE_DRAW_CONNECTIONS) {
      const a = results.poseLandmarks[i];
      const b = results.poseLandmarks[j];
      if (!a || !b) continue;
      canvasCtx.beginPath();
      canvasCtx.moveTo(a.x * width, a.y * height);
      canvasCtx.lineTo(b.x * width, b.y * height);
      canvasCtx.stroke();
    }
    canvasCtx.fillStyle = "#00FF00";
    for (const idx of POSE_INDICES) {
      const lm = results.poseLandmarks[idx];
      if (!lm) continue;
      canvasCtx.beginPath();
      canvasCtx.arc(lm.x * width, lm.y * height, 4, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  }

  // Left hand (red)
  if (results.leftHandLandmarks) {
    canvasCtx.fillStyle = "#FF0000";
    for (const lm of results.leftHandLandmarks) {
      canvasCtx.beginPath();
      canvasCtx.arc(lm.x * width, lm.y * height, 2, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  }

  // Right hand (blue)
  if (results.rightHandLandmarks) {
    canvasCtx.fillStyle = "#0000FF";
    for (const lm of results.rightHandLandmarks) {
      canvasCtx.beginPath();
      canvasCtx.arc(lm.x * width, lm.y * height, 2, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  }

  // Face (yellow)
  if (results.faceLandmarks) {
    canvasCtx.fillStyle = "#FFFF00";
    for (const idx of FACE_INDICES) {
      const lm = results.faceLandmarks[idx];
      if (!lm) continue;
      canvasCtx.beginPath();
      canvasCtx.arc(lm.x * width, lm.y * height, 2, 0, 2 * Math.PI);
      canvasCtx.fill();
    }
  }
}

export default function useMediaPipe(videoRef, canvasRef) {
  const [isReady, setIsReady] = useState(false);
  const [error, setError] = useState(null);

  const landmarkerRef = useRef(null);
  const animFrameRef = useRef(null);
  const streamRef = useRef(null);
  const onResultsRef = useRef(null);

  // 외부에서 프레임 결과 콜백 등록
  const setOnResults = useCallback((callback) => {
    onResultsRef.current = callback;
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const videoEl = videoRef.current;
        const canvasEl = canvasRef.current;
        if (!videoEl || !canvasEl) return;

        const canvasCtx = canvasEl.getContext("2d");

        // CDN에서 Tasks API 로드
        const { HolisticLandmarker, FilesetResolver } =
          await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/+esm");

        if (cancelled) return;

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm",
        );

        if (cancelled) return;

        const landmarker = await HolisticLandmarker.createFromOptions(vision, {
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

        if (cancelled) return;
        landmarkerRef.current = landmarker;

        // 웹캠 열기
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, frameRate: { ideal: 30, max: 30 } },
        });

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;
        videoEl.srcObject = stream;
        await new Promise((resolve) => {
          videoEl.onloadedmetadata = resolve;
        });
        videoEl.play();

        // 감지 루프 (rAF ~30fps)
        function detectLoop(now) {
          if (cancelled) return;
          animFrameRef.current = requestAnimationFrame(detectLoop);
          if (videoEl.readyState < 2) return;

          canvasEl.width = videoEl.videoWidth || 480;
          canvasEl.height = videoEl.videoHeight || 360;

          const result = landmarker.detectForVideo(videoEl, now);
          const adapted = adaptResult(result);

          drawOnCanvas(canvasCtx, adapted, canvasEl.width, canvasEl.height);

          if (onResultsRef.current) {
            onResultsRef.current(adapted);
          }
        }

        setIsReady(true);
        detectLoop(performance.now());
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
          console.error("MediaPipe init error:", err);
        }
      }
    }

    init();

    // cleanup
    return () => {
      cancelled = true;
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (landmarkerRef.current) {
        landmarkerRef.current.close();
      }
    };
  }, [videoRef, canvasRef]);

  return { isReady, error, setOnResults };
}
