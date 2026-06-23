/**
 * useSignRecorder — 수어 키포인트 녹화 & /predict SSE 전송
 *
 * - startCountdown() 호출 → 3초 카운트다운 후 녹화 시작
 * - stopRecording() 호출 → 즉시 종료 후 POST /predict → SSE 수신
 * - 손이 HAND_ABSENT_THRESHOLD 프레임 이상 미감지 시 단어 경계 확정
 * - 단어별 세그먼트를 즉시 분리해서 관리 (demo 동일 방식)
 */

import { useState, useRef, useCallback, useEffect } from "react";
import { extractKeypoints, framesToPredictBodyMulti } from "../utils/keypoints";

const RECORD_INTERVAL_MS = 1000 / 60;
const MIN_FRAMES = Number(import.meta.env.VITE_MIN_FRAMES) || 10;
const HAND_ABSENT_THRESHOLD = Number(import.meta.env.VITE_PAUSE_PERIOD) || 15;
const COUNTDOWN_SECS = 2;

export default function useSignRecorder(apiBaseUrl) {
  const [isRecording, setIsRecording] = useState(false);
  const [frameCount, setFrameCount] = useState(0);
  const [wordCount, setWordCount] = useState(0);
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [countdownProgress, setCountdownProgress] = useState(0);
  const [status, setStatus] = useState("idle");
  const [statusText, setStatusText] = useState("버튼을 눌러 수어를 시작하세요");
  const [translatedText, setTranslatedText] = useState("");
  const [sseEvents, setSseEvents] = useState([]);

  // 녹화 버퍼 (demo 방식: 단어별 즉시 분리)
  const currentSegmentRef = useRef([]); // 현재 수집 중인 단어 프레임
  const wordSegmentsRef   = useRef([]); // 완성된 단어 세그먼트 배열
  const hiddenCountRef    = useRef(0);  // 손 미감지 연속 프레임 수
  const lastTsRef         = useRef(0);
  const recordingRef      = useRef(false);

  // 카운트다운
  const countingDownRef = useRef(false);
  const countdownRafRef = useRef(null);

  // ── /predict 전송 (SSE 스트리밍) ───────────────────────────────────
  const sendPredict = useCallback(
    async (segments) => {
      if (segments.length === 0) {
        setStatus("error");
        setStatusText("프레임이 너무 적습니다. 더 길게 수어를 시도하세요.");
        return;
      }

      // segments → flat frames + boundaries (demo sendForRecognition 방식)
      const allFrames = segments.flat();
      const boundaries = [];
      let cumulative = 0;
      for (let i = 0; i < segments.length - 1; i++) {
        cumulative += segments[i].length;
        boundaries.push(cumulative);
      }

      const body = framesToPredictBodyMulti(allFrames, boundaries, MIN_FRAMES);
      setStatus("processing");
      setStatusText(`${allFrames.length}프레임 (${segments.length}단어) 번역 중...`);

      try {
        const res = await fetch(`${apiBaseUrl}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let sentence = "";
        let sawDone = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;

            try {
              const event = JSON.parse(jsonStr);
              setSseEvents((prev) => [...prev, event]);

              switch (event.status) {
                case "received":
                  setStatusText(`서버 수신 완료 (${event.content.n_words}개 단어)`);
                  break;
                case "rag-retrieve":
                  setStatusText(`RAG 검색 중... (${event.content.idx + 1}/${event.content.total})`);
                  break;
                case "generate-sentence":
                  setStatusText("문장 생성 중...");
                  break;
                case "sentence-chunk":
                  sentence += event.content.chunk;
                  setTranslatedText(sentence);
                  setStatusText("번역 중...");
                  break;
                case "error":
                  console.warn("SSE error:", event.content);
                  break;
                case "done":
                  sawDone = true;
                  setStatus("done");
                  setStatusText("번역 완료");
                  break;
                default:
                  break;
              }
            } catch {
              // JSON 파싱 실패 — 무시
            }
          }
        }

        if (!sawDone) {
          setStatus("done");
          setStatusText("번역 완료");
        }
      } catch (err) {
        setStatus("error");
        setStatusText(`오류: ${err.message}`);
        console.error("predict error:", err);
      }
    },
    [apiBaseUrl],
  );

  // ── 녹화 시작 ──────────────────────────────────────────────────────
  const startRecording = useCallback(() => {
    currentSegmentRef.current = [];
    wordSegmentsRef.current = [];
    hiddenCountRef.current = 0;
    lastTsRef.current = 0;
    recordingRef.current = true;
    setIsRecording(true);
    setFrameCount(0);
    setWordCount(0);
    setStatus("recording");
    setStatusText("수어 인식 중... (손이 사라지면 단어가 구분됩니다)");
    setTranslatedText("");
    setSseEvents([]);
  }, []);

  // ── 녹화 종료 → /predict 전송 ──────────────────────────────────────
  const stopRecording = useCallback(() => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    setIsRecording(false);

    // 마지막 세그먼트 저장 (demo 동일)
    if (currentSegmentRef.current.length >= MIN_FRAMES) {
      wordSegmentsRef.current.push([...currentSegmentRef.current]);
    }

    const segments = [...wordSegmentsRef.current];
    currentSegmentRef.current = [];
    wordSegmentsRef.current = [];

    console.log(`[녹화] ${segments.length}개 단어, 세그먼트별 프레임:`, segments.map(s => s.length));
    sendPredict(segments);
  }, [sendPredict]);

  // ── 버튼 클릭 → 3초 카운트다운 → 녹화 시작 ────────────────────────
  const startCountdown = useCallback(() => {
    if (countingDownRef.current || recordingRef.current) return;
    countingDownRef.current = true;
    setIsCountingDown(true);
    setCountdownProgress(0);
    setStatus("countdown");
    setStatusText("곧 녹화를 시작합니다...");

    const start = performance.now();
    const tick = () => {
      const p = Math.min(1, (performance.now() - start) / (COUNTDOWN_SECS * 1000));
      setCountdownProgress(p);
      if (p < 1) {
        countdownRafRef.current = requestAnimationFrame(tick);
      } else {
        countingDownRef.current = false;
        setIsCountingDown(false);
        startRecording();
      }
    };
    countdownRafRef.current = requestAnimationFrame(tick);
  }, [startRecording]);

  useEffect(
    () => () => {
      if (countdownRafRef.current) cancelAnimationFrame(countdownRafRef.current);
    },
    [],
  );

  // ── MediaPipe 프레임 콜백 (demo _pushFrame + _trackWordBoundary 동일) ─
  const onFrame = useCallback((results) => {
    const now = performance.now();
    if (now - lastTsRef.current < RECORD_INTERVAL_MS) return;
    lastTsRef.current = now;

    if (!recordingRef.current) return;

    const frame = extractKeypoints(results);
    const handVisible = !!(frame.leftHand || frame.rightHand);

    if (handVisible) {
      currentSegmentRef.current.push(frame);
      const total = wordSegmentsRef.current.reduce((s, seg) => s + seg.length, 0)
                  + currentSegmentRef.current.length;
      setFrameCount(total);
    }

    // 손 미감지 기반 단어 경계 (demo _trackWordBoundary 동일)
    if (!handVisible) {
      hiddenCountRef.current += 1;
      if (hiddenCountRef.current === HAND_ABSENT_THRESHOLD) {
        if (currentSegmentRef.current.length >= MIN_FRAMES) {
          wordSegmentsRef.current.push([...currentSegmentRef.current]);
          const wc = wordSegmentsRef.current.length;
          setWordCount(wc);
          setStatusText(`단어 ${wc}번째 완료 — 다음 단어로 넘어가세요`);
        }
        currentSegmentRef.current = [];
      }
    } else {
      if (hiddenCountRef.current >= HAND_ABSENT_THRESHOLD) {
        setStatusText(`단어 ${wordSegmentsRef.current.length + 1}번째 녹화 중...`);
      }
      hiddenCountRef.current = 0;
    }
  }, []);

  const resetStatus = useCallback(() => {
    setStatus("idle");
    setStatusText("버튼을 눌러 수어를 시작하세요");
    setTranslatedText("");
    setSseEvents([]);
    setFrameCount(0);
    setWordCount(0);
  }, []);

  return {
    isRecording,
    frameCount,
    wordCount,
    isCountingDown,
    countdownProgress,
    status,
    statusText,
    translatedText,
    sseEvents,
    onFrame,
    startCountdown,
    stopRecording,
    resetStatus,
  };
}
