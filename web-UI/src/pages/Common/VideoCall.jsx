import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import api from '../../api/index.js';
import useWebRTC from '../../hooks/useWebRTC.js';
import useSignRecorder from '../../hooks/useSignRecorder.js';
import {
  adaptResult,
  POSE_INDICES,
  FACE_INDICES,
  POSE_DRAW_CONNECTIONS,
} from '../../utils/keypoints.js';
import './VideoCall.css';

const COUNTDOWN_SECS = 2;
const TEST_MODE = import.meta.env.VITE_TEST_MODE === 'true';
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const COUNTDOWN_RING_R = 52;
const COUNTDOWN_RING_CIRC = 2 * Math.PI * COUNTDOWN_RING_R;


function drawLandmarks(ctx, results, w, h, vw = 0, vh = 0) {
  ctx.clearRect(0, 0, w, h);
  let dispW = w, dispH = h, offX = 0, offY = 0;
  if (vw > 0 && vh > 0) {
    const scale = Math.max(w / vw, h / vh);
    dispW = vw * scale;
    dispH = vh * scale;
    offX = (w - dispW) / 2;
    offY = (h - dispH) / 2;
  }
  const mapX = (nx) => offX + nx * dispW;
  const mapY = (ny) => offY + ny * dispH;

  if (results.poseLandmarks) {
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth = 2;
    for (const [i, j] of POSE_DRAW_CONNECTIONS) {
      const a = results.poseLandmarks[i];
      const b = results.poseLandmarks[j];
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(mapX(a.x), mapY(a.y));
      ctx.lineTo(mapX(b.x), mapY(b.y));
      ctx.stroke();
    }
    ctx.fillStyle = '#00FF00';
    for (const idx of POSE_INDICES) {
      const lm = results.poseLandmarks[idx];
      if (!lm) continue;
      ctx.beginPath();
      ctx.arc(mapX(lm.x), mapY(lm.y), 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  if (results.leftHandLandmarks) {
    ctx.fillStyle = '#FF0000';
    for (const lm of results.leftHandLandmarks) {
      ctx.beginPath();
      ctx.arc(mapX(lm.x), mapY(lm.y), 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  if (results.rightHandLandmarks) {
    ctx.fillStyle = '#0000FF';
    for (const lm of results.rightHandLandmarks) {
      ctx.beginPath();
      ctx.arc(mapX(lm.x), mapY(lm.y), 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  if (results.faceLandmarks) {
    ctx.fillStyle = '#FFFF00';
    for (const idx of FACE_INDICES) {
      const lm = results.faceLandmarks[idx];
      if (!lm) continue;
      ctx.beginPath();
      ctx.arc(mapX(lm.x), mapY(lm.y), 2, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function VideoCall() {
  const navigate = useNavigate();
  const { id } = useParams();
  const role = localStorage.getItem('role');
  const isPatient = role === 'patient';

  const [isMuted, setIsMuted] = useState(false);
  const [isCameraOff, setIsCameraOff] = useState(false);
  const [translationText, setTranslationText] = useState('');
  const [responseText, setResponseText] = useState('');
  const [mpReady, setMpReady] = useState(false);
  const [turn, setTurn] = useState('doctor'); // 'doctor' | 'patient'
  const [chatHistory, setChatHistory] = useState([]);
  const [showConnected, setShowConnected] = useState(false);

  const myVideoRef = useRef(null);
  const remoteMainRef = useRef(null);
  const remoteSmallRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const animFrameRef = useRef(null);
  const recognitionRef = useRef(null);
  const chatIdRef = useRef(0);
  const chatEndRef = useRef(null);
  const doctorSpeechRef = useRef('');

  const {
    isRecording, frameCount, wordCount,
    isCountingDown, countdownProgress,
    status: recStatus, statusText: recStatusText,
    translatedText: signTranslation,
    onFrame, startCountdown, stopRecording, resetStatus,
  } = useSignRecorder(API_BASE_URL);

  const appendChat = useCallback((role, text) => {
    if (!text?.trim()) return;
    setChatHistory(prev => [...prev, { id: chatIdRef.current++, role, text: text.trim() }]);
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleTranslation = useCallback((text) => {
    setTranslationText(text);
    if (!isPatient) appendChat('patient', text);
  }, [isPatient, appendChat]);

  const handleSTTResult = useCallback((text) => {
    setResponseText(text);
    if (isPatient) appendChat('doctor', text);
  }, [isPatient, appendChat]);

  const handleSignToggle = () => {
    if (!mpReady || recStatus === 'processing' || turn !== 'patient') return;
    if (isRecording) stopRecording();
    else if (!isCountingDown) startCountdown();
  };

  const handleTurnChange = useCallback((newTurn) => {
    setTurn(newTurn);
    if (newTurn === 'patient') resetStatus();
  }, [resetStatus]);

  const handlePeerLeft = useCallback(() => {
    if (isPatient) navigate(`/patient/prescription/${id}`);
    else navigate(`/doctor/prescription/${id}`);
  }, [isPatient, id, navigate]);

  const { localStream, remoteStream, isConnected, sendWS } = useWebRTC({
    consultationId: id, role,
    onTranslation: handleTranslation,
    onSTTResult: handleSTTResult,
    onPeerLeft: handlePeerLeft,
    onTurnChange: handleTurnChange,
  });

  const handleDoctorDone = useCallback(() => {
    const speech = doctorSpeechRef.current.trim();
    if(speech) {
      appendChat('doctor', speech);
      sendWS({ type: 'STT_DONE', text: speech });
    }
    doctorSpeechRef.current = '';
    setResponseText('');
    resetStatus();
    setTurn('patient');
    sendWS({ type: 'TURN_CHANGE', turn: 'patient' });
  }, [sendWS, resetStatus, appendChat]);

  useEffect(() => {
    const videoEl = myVideoRef.current;
    if (!videoEl) return;
    let cancelled = false;
    let fallbackStream = null;
    async function attachStream() {
      let stream = localStream;
      if (!stream) {
        await new Promise((r) => setTimeout(r, 800));
        if (cancelled) return;
        try {
          fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          stream = fallbackStream;
        } catch (err) { console.error('카메라 fallback 실패:', err); return; }
      }
      if (cancelled || !stream) { fallbackStream?.getTracks().forEach((t) => t.stop()); return; }
      if (videoEl.srcObject !== stream) videoEl.srcObject = stream;
      try { await videoEl.play(); } catch {}
    }
    attachStream();
    return () => { cancelled = true; if (fallbackStream) fallbackStream.getTracks().forEach((t) => t.stop()); };
  }, [localStream]);

  useEffect(() => {
    if (remoteMainRef.current) remoteMainRef.current.srcObject = remoteStream;
    if (remoteSmallRef.current) remoteSmallRef.current.srcObject = remoteStream;
  }, [remoteStream]);

  useEffect(() => {
    if (id && role === 'doctor') api.patch(`/api/consultations/${id}/start`).catch(() => {});
  }, [id, role]);

  // 실시간 표시 (SSE chunk 누적)
  useEffect(() => {
    if (signTranslation && isPatient) {
      setTranslationText(signTranslation);
    }
  }, [signTranslation, isPatient]);

  // 번역 완료 시점에만 chat 추가 + 의사에게 전송 (chunk마다 중복 박스 생성 방지)
  useEffect(() => {
    if (recStatus === 'done' && signTranslation && isPatient) {
      appendChat('patient', signTranslation);
      try { sendWS({ type: 'TRANSLATION_RESULT', result: signTranslation }); } catch (e) {}
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recStatus, isPatient]);

  // 환자 턴 종료: 번역 완료 시 의사 턴으로 전환
  useEffect(() => {
    if (recStatus === 'done' && isPatient && turn === 'patient') {
      setTurn('doctor');
      sendWS({ type: 'TURN_CHANGE', turn: 'doctor' });
    }
  }, [recStatus, isPatient, turn, sendWS]);

  useEffect(() => {
    if (!isPatient) return;
    let cancelled = false;
    let lastTs = 0;
    async function initMediaPipe() {
      try {
        const { HolisticLandmarker, FilesetResolver } = await import(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/+esm'
        );
        if (cancelled) return;
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.15/wasm'
        );
        if (cancelled) return;
        const landmarker = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task',
            delegate: 'CPU',
          },
          runningMode: 'VIDEO',
          minFaceDetectionConfidence: 0.5, minFaceSuppressionThreshold: 0.5, minFaceLandmarksConfidence: 0.5,
          minPoseDetectionConfidence: 0.5, minPoseSuppressionThreshold: 0.5, minPoseLandmarksConfidence: 0.5,
          minHandLandmarksConfidence: 0.5,
          outputFaceBlendshapes: false, outputSegmentationMasks: false,
        });
        if (cancelled) return;
        landmarkerRef.current = landmarker;
        setMpReady(true);
        const canvasEl = canvasRef.current;
        const ctx = canvasEl?.getContext('2d');
        function detectLoop() {
          if (cancelled) return;
          animFrameRef.current = requestAnimationFrame(detectLoop);
          const videoEl = myVideoRef.current;
          if (!videoEl || videoEl.readyState < 2) return;
          if (canvasEl) {
            const cw = canvasEl.clientWidth || videoEl.videoWidth || 480;
            const ch = canvasEl.clientHeight || videoEl.videoHeight || 360;
            if (canvasEl.width !== cw) canvasEl.width = cw;
            if (canvasEl.height !== ch) canvasEl.height = ch;
          }
          let ts = performance.now();
          if (ts <= lastTs) ts = lastTs + 1;
          lastTs = ts;
          let result;
          try { result = landmarker.detectForVideo(videoEl, ts); } catch (err) { console.error('detectForVideo:', err); return; }
          const adapted = adaptResult(result);
          if (ctx) drawLandmarks(ctx, adapted, canvasEl.width, canvasEl.height, videoEl.videoWidth, videoEl.videoHeight);
          onFrame(adapted);
        }
        detectLoop();
      } catch (err) { console.error('MediaPipe init error:', err); }
    }
    initMediaPipe();
    return () => {
      cancelled = true;
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
      if (landmarkerRef.current) { landmarkerRef.current.close(); landmarkerRef.current = null; }
    };
  }, [isPatient, onFrame]);

  // 의사 턴일 때만 음성 인식 활성화
  useEffect(() => {
    if (isPatient || turn !== 'doctor') {
      if (recognitionRef.current) { recognitionRef.current.stop(); recognitionRef.current = null; }
      return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    let stopped = false;
    const startRecognition = () => {
      if (stopped) return;
      const recognition = new SpeechRecognition();
      recognition.lang = 'ko-KR'; recognition.continuous = true; recognition.interimResults = true;
      recognition.onresult = (event) => {
        let interim = '', final = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const t = event.results[i][0].transcript;
          if (event.results[i].isFinal) final += t; else interim += t;
        }
        if (final) {
          doctorSpeechRef.current += final;
        }
        setResponseText(doctorSpeechRef.current + interim);
      };
      recognition.onerror = (e) => { if (e.error !== 'not-allowed' && e.error !== 'service-not-allowed') recognition.stop(); };
      recognition.onend = () => { if (!stopped) startRecognition(); };
      recognition.start();
      recognitionRef.current = recognition;
    };
    startRecognition();
    return () => {
      stopped = true;
      if (recognitionRef.current) { recognitionRef.current.stop(); recognitionRef.current = null; }
    };
  }, [isPatient, turn, sendWS, appendChat]);

  useEffect(() => {
    if (isConnected) {
      setShowConnected(true);
      setTimeout(() => setShowConnected(false), 3000);
    }
  }, [isConnected]);

  const handleEnd = async () => {
    if (id) { try { await api.patch(`/api/consultations/${id}/end`); } catch (e) {} }
    if (localStream) localStream.getTracks().forEach((t) => t.stop());
    if (recognitionRef.current) recognitionRef.current.stop();
    navigate(isPatient ? `/patient/prescription/${id}` : `/doctor/prescription/${id}`);
  };

  const handleCameraToggle = () => {
    if (localStream) {
      localStream.getVideoTracks().forEach((t) => { t.enabled = !t.enabled; });
      if (myVideoRef.current) myVideoRef.current.style.display = isCameraOff ? 'block' : 'none';
      setIsCameraOff(!isCameraOff);
    }
  };

  return (
    <div className="vc-root">
      <div className="vc-header" style={{ backgroundColor: isPatient ? '#2563EB' : '#34A853', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
        <h2 className="vc-header-title">{isPatient && !isConnected ? '대기 중' : '진료 중'}</h2>
        <button onClick={handleEnd} style={{ position: 'absolute', right: '16px', background: '#EF4444', border: 'none', borderRadius: '20px', padding: '6px 14px', color: '#fff', fontSize: '14px', fontWeight: '600', cursor: 'pointer' }}>
          종료
        </button>
      </div>
      {showConnected && ( 
        <div style={{ position: 'fixed', top: '70px', left: '50%', transform: 'translateX(-50%)', background: '#22c55e', color: '#fff', padding: '10px 24px', borderRadius: '24px', fontWeight: '700', fontSize: '15px', zIndex: 999 }}>
        ✅ 연결됐습니다!
        </div>
      )}
      <div className="vc-main">
        <div className="vc-left" style={{ display: turn === 'patient' ? 'flex' : 'none', flex: 1, minWidth: 0 }}>
          <div className="vc-camera-area">
            {isPatient ? (
              <div className="vc-video-container" style={{ position: (turn === 'patient') ? 'relative' : 'absolute', visibility: (turn === 'patient') ? 'visible' : 'hidden', width: '100%', height: '100%' }}>
                <video ref={myVideoRef} autoPlay muted playsInline className="vc-main-video mirrored" />
                <canvas ref={canvasRef} className="vc-landmark-canvas" />
              </div>
            ) : (
              <video ref={remoteMainRef} autoPlay playsInline className="vc-main-video" style={{ backgroundColor: '#0F172A', visibility: turn === 'patient' ? 'visible' : 'hidden' }} />
            )}
            {isPatient && !isConnected && (
              <div className="vc-waiting-overlay">
                <div style={{ fontSize: '40px', marginBottom: '12px' }}>🩺</div>
                <div className="vc-waiting-title">의사 입장 대기 중</div>
                <div className="vc-waiting-sub">잠시만 기다려주세요</div>
              </div>
            )}
            {isRecording && (
              <div className="vc-rec-indicator"><span className="vc-rec-dot" />REC · {frameCount}f</div>
            )}
            {isPatient && isCountingDown && (
              <div className="vc-countdown-overlay">
                <div className="vc-countdown-ring-wrap">
                  <svg className="vc-countdown-ring" viewBox="0 0 120 120">
                    <circle className="vc-countdown-track" cx="60" cy="60" r={COUNTDOWN_RING_R} />
                    <circle className="vc-countdown-progress" cx="60" cy="60" r={COUNTDOWN_RING_R}
                      style={{ strokeDasharray: COUNTDOWN_RING_CIRC, strokeDashoffset: COUNTDOWN_RING_CIRC * (1 - countdownProgress) }} />
                  </svg>
                  <div className="vc-countdown-num">{Math.max(1, Math.ceil((1 - countdownProgress) * COUNTDOWN_SECS))}</div>
                </div>
                <div className="vc-countdown-cap">곧 녹화를 시작합니다</div>
              </div>
            )}
            <div className="vc-pip" style={{ borderColor: isPatient ? '#1986DC' : '#34A853' }}>
              {isPatient ? (
                <video ref={remoteSmallRef} autoPlay playsInline className="vc-pip-video" />
              ) : (
                <video ref={myVideoRef} autoPlay muted playsInline className="vc-pip-video mirrored" />
              )}
            </div>
          </div>
        </div>
        <div className="vc-right" style={{ width: turn === 'doctor' ? '100%' : '420px', flexShrink: 0 }}>
          <div className="vc-right-inner">
            {isPatient && (
              <div
                className={`vc-sign-btn ${isRecording ? 'recording' : ''} ${turn !== 'patient' ? 'disabled' : ''}`}
                style={{ textAlign: 'center', cursor: (!mpReady || recStatus === 'processing' || isCountingDown || turn !== 'patient') ? 'not-allowed' : 'pointer', opacity: turn !== 'patient' ? 0.4 : 1 }}
                onClick={handleSignToggle}
              >
                {turn !== 'patient' ? '🔒 의사 발화 중...'
                  : !mpReady ? '⏳ MediaPipe 로딩 중...'
                  : recStatus === 'processing' ? '⌛ 번역 중...'
                  : isCountingDown ? '⏱ 곧 시작합니다...'
                  : isRecording ? `🟢 수어 인식 중 · 단어 ${wordCount + 1} · ${frameCount}f`
                  : '▶ 수어 시작'}
              </div>
            )}
            {!isPatient && turn === 'doctor' && (
              <div className="vc-doctor-done-btn" onClick={handleDoctorDone}>
                발화 완료 — 환자 턴으로 넘기기
              </div>
            )}
            {isPatient && recStatusText && <p className={`vc-rec-status ${recStatus}`}>{recStatusText}</p>}
            <div className="vc-chat-area">
              {chatHistory.length === 0 && (
                <p className="vc-chat-empty">대화를 시작하세요</p>
              )}
              {chatHistory.map(msg => {
                const isSign = msg.role === 'patient';
                const color = isSign ? '#2563EB' : '#34A853';
                const label = isSign
                  ? (isPatient ? '내 수어 번역 결과' : '환자 수어 번역')
                  : (isPatient ? '의사 음성 결과' : '의사 음성 번역');
                return (
                  <div key={msg.id} className="vc-result-box" style={{ borderColor: color }}>
                    <span className="vc-result-badge" style={{ backgroundColor: color }}>{label}</span>
                    <p className="vc-result-text" style={{ margin: 0 }}>{msg.text}</p>
                  </div>
                );
              })}
              {responseText && turn === 'doctor' && !isPatient && (
                <div className="vc-result-box" style={{ borderColor: '#34A853' }}>
                  <span className="vc-result-badge" style={{ backgroundColor: '#34A853' }}>내 음성 (입력 중)</span>
                  <p className="vc-result-text" style={{ margin: 0 }}>{responseText}</p>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VideoCall;