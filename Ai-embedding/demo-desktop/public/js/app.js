/**
 * Main application logic.
 *
 * Auto mode:
 *   화면 상단 버튼 존(y < TRIGGER_ZONE_Y)에 손이 TRIGGER_DWELL 연속 유지 →
 *   시작 또는 종료 토글. 드웰 중 프레임은 버퍼에 넣지 않음.
 *
 * Manual mode: Record / Stop 버튼.
 */

// ── Config ──────────────────────────────────────────────────────────────────
const TRIGGER_ZONE_Y   = 0.80;   // y > 이 값: 화면 하단
const TRIGGER_ZONE_X   = 0.20;   // x < 이 값: 미러 기준 화면 우측 (MediaPipe 좌측, 80px 버튼 기준)
const TRIGGER_DWELL    = 8;      // 연속 프레임 (~0.5 s @ 15 fps)
const TRIGGER_COOLDOWN = 20;     // 트리거 후 무시 프레임 수 (~1.3 s)
const MIN_FRAMES       = 10;

// ── State ────────────────────────────────────────────────────────────────────
let isRecording    = false;
let dwellCount     = 0;
let cooldownCount  = 0;
let serverConfig   = null;
let autoMode       = true;

// 손 가시성 기반 단어 세그먼트
let wordSegments   = [];   // 완성된 단어 세그먼트 배열 [[frame,...], ...]
let currentSegment = [];   // 현재 수집 중인 단어 프레임
let wordCount      = 0;    // 완성된 단어 수

// 녹화 샘플링 (60fps)
const RECORD_INTERVAL_MS = 1000 / 60;
let _lastRecordTs = 0;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const btnRecord    = document.getElementById("btn-record");
const btnStop      = document.getElementById("btn-stop");
const chkAuto      = document.getElementById("chk-auto");
const chkWordMode  = document.getElementById("chk-word-mode");
const handStatus   = document.getElementById("hand-status");
const handIcon     = document.getElementById("hand-icon");
const handLabel    = document.getElementById("hand-label");
const recIndicator = document.getElementById("rec-indicator");
const triggerBtn   = document.getElementById("trigger-btn");
const triggerLabel = document.getElementById("trigger-label");
const triggerBar   = document.getElementById("trigger-bar");
const startOverlay  = document.getElementById("start-overlay");
const reqCountEl    = document.getElementById("req-count");
const countdownRing = document.querySelector(".countdown-ring");
const COUNTDOWN_C   = 2 * Math.PI * 22;  // circumference for r=22

// ── Start countdown ───────────────────────────────────────────────────────────
function _showCountdown() {
  return new Promise(resolve => {
    // Reset ring
    countdownRing.style.transition = "none";
    countdownRing.style.strokeDasharray  = String(COUNTDOWN_C);
    countdownRing.style.strokeDashoffset = String(COUNTDOWN_C);
    startOverlay.classList.remove("hidden");

    // Force reflow so the reset is applied before the animation starts
    void countdownRing.getBoundingClientRect();

    // Animate fill over 1 second
    countdownRing.style.transition = "stroke-dashoffset 1s linear";
    countdownRing.style.strokeDashoffset = "0";

    setTimeout(() => {
      startOverlay.classList.add("hidden");
      countdownRing.style.transition = "none";
      resolve();
    }, 1000);
  });
}

// ── Server config ─────────────────────────────────────────────────────────────
async function fetchServerConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    serverConfig = await res.json();
    setServerInfo(serverConfig);
    return serverConfig;
  } catch (e) {
    setStatus("서버에 연결할 수 없습니다. FastAPI 서버를 먼저 시작하세요.", "error");
    return null;
  }
}

// ── Recording control ─────────────────────────────────────────────────────────
function startRecording() {
  wordSegments    = [];
  currentSegment  = [];
  wordCount       = 0;
  dwellCount      = 0;
  _handWasVisible = false;
  _hiddenCount    = 0;
  _lastRecordTs   = 0;
  isRecording     = true;
  recIndicator.classList.remove("hidden");
  setFrameCount(0);
  reqCountEl.textContent = "감지 단어: 0개";
  triggerLabel.textContent = "■";
  triggerBtn.classList.add("recording-active");

  if (!autoMode) {
    btnRecord.disabled = true;
    btnRecord.classList.add("recording");
    btnStop.disabled = false;
  }
  setStatus("녹화 중...", "recording");
  _setHandUI(true);
}

async function stopRecording() {
  isRecording = false;
  dwellCount  = 0;
  recIndicator.classList.add("hidden");
  triggerLabel.textContent = "▶";
  triggerBtn.classList.remove("recording-active");
  triggerBar.style.background = "#27272a";

  if (!autoMode) {
    btnRecord.disabled = false;
    btnRecord.classList.remove("recording");
    btnStop.disabled = true;
  }
  _setHandUI(false);

  // 마지막 세그먼트가 아직 남아있으면 저장
  if (currentSegment.length >= MIN_FRAMES) {
    wordSegments.push([...currentSegment]);
    wordCount++;
  }
  currentSegment = [];

  const totalFrames = wordSegments.reduce((s, seg) => s + seg.length, 0);
  if (wordSegments.length === 0 || totalFrames < MIN_FRAMES) {
    setStatus("프레임이 너무 적습니다. 더 길게 수화를 시도하세요.", "error");
    return;
  }

  setStatus(`${wordSegments.length}개 단어 (${totalFrames}프레임) 인식 중...`, "processing");
  await sendForRecognition(wordSegments);
  wordSegments = [];
}

// ── API call ──────────────────────────────────────────────────────────────────
async function sendForRecognition(segments) {
  try {
    // segments: [[frame,...], [frame,...], ...] — 단어별 분리된 세그먼트
    const allFrames = segments.flat();
    const boundaries = [];
    let cumulative = 0;
    for (let i = 0; i < segments.length - 1; i++) {
      cumulative += segments[i].length;
      boundaries.push(cumulative);
    }
    console.log(`[predict] ${segments.length}개 단어, 세그먼트별 프레임:`, segments.map(s => s.length), '경계:', boundaries);
    const res = await fetch("/api/recognize/sentence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keypoints: allFrames.map(flattenKeypoints), word_mode: chkWordMode.checked, boundaries }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    showSentenceResults(data);
    setStatus("완료 — 버튼에 손을 올리면 다시 시작됩니다", "ready");
  } catch (e) {
    setStatus(`오류: ${e.message}`, "error");
  } finally {
    _setHandUI(false);
  }
}

// ── 실시간 단어 경계 감지 (손 가시성 기반, 15프레임 유예) ──────────────────
const HAND_ABSENT_THRESHOLD = 15;  // 손이 없는 프레임이 이 수를 넘으면 단어 구분
let _handWasVisible = false;
let _hiddenCount    = 0;

function _trackWordBoundary(handVisible) {
  if (!handVisible) {
    _hiddenCount++;
    // 15프레임 초과 → 단어 경계 확정
    if (_hiddenCount === HAND_ABSENT_THRESHOLD) {
      if (currentSegment.length >= MIN_FRAMES) {
        wordSegments.push([...currentSegment]);
        wordCount++;
        reqCountEl.textContent = `감지 단어: ${wordCount}개`;
        setStatus(`단어 ${wordCount}번째 완료 — 다음 단어로 넘어가세요`, "processing");
      }
      currentSegment = [];
    }
  } else {
    if (_hiddenCount >= HAND_ABSENT_THRESHOLD) {
      // 확정된 경계 이후 손 재등장 → 새 단어 시작
      setStatus(`단어 ${wordCount + 1}번째 녹화 중...`, "recording");
    }
    _hiddenCount = 0;
  }
  _handWasVisible = handVisible;
}

function _pushFrame(results) {
  const now = performance.now();
  if (now - _lastRecordTs < RECORD_INTERVAL_MS) return;
  _lastRecordTs = now;

  const frame = extractKeypoints(results);
  const handVisible = !!(frame.leftHand || frame.rightHand);

  if (handVisible) {
    currentSegment.push(frame);
    const totalFrames = wordSegments.reduce((s, seg) => s + seg.length, 0) + currentSegment.length;
    setFrameCount(totalFrames);
  }

  _trackWordBoundary(handVisible);
}

// ── Hand in trigger zone detection ───────────────────────────────────────────
function _isHandInZone(results) {
  const check = (lms) => {
    if (!lms || lms.length === 0) return false;
    // 하나라도 우측 하단 박스 안에 들어오면 인식
    return lms.some(lm => lm.y > TRIGGER_ZONE_Y && lm.x < TRIGGER_ZONE_X);
  };
  return check(results.leftHandLandmarks) || check(results.rightHandLandmarks);
}

// ── MediaPipe callback ────────────────────────────────────────────────────────
function onMediaPipeResults(results) {
  if (typeof window._recordingHook === "function") window._recordingHook(results);
  if (typeof window._dbRecordingHook === "function") window._dbRecordingHook(results);

  if (window.currentTab === "db" || window.currentTab === "test") return;

  if (autoMode) {
    _handleAutoMode(results);
  } else {
    if (isRecording) _pushFrame(results);
  }
}

function _handleAutoMode(results) {
  // 쿨다운: 버튼 존 무시, 녹화 중이면 프레임 수집
  if (cooldownCount > 0) {
    cooldownCount--;
    if (isRecording) _pushFrame(results);
    return;
  }

  const inZone = _isHandInZone(results);

  if (inZone) {
    dwellCount++;
    const pct      = Math.min(100, (dwellCount / TRIGGER_DWELL) * 100);
    const barColor = isRecording ? "#ef4444" : "#3b82f6";
    triggerBar.style.background =
      `linear-gradient(to bottom, ${barColor} ${pct}%, #27272a ${pct}%)`;
    triggerBtn.classList.add("hand-near");

    if (dwellCount >= TRIGGER_DWELL) {
      // 드웰 달성 → 토글
      dwellCount    = 0;
      cooldownCount = TRIGGER_COOLDOWN;
      triggerBar.style.background = "#27272a";
      triggerBtn.classList.remove("hand-near");
      if (!isRecording) {
        _showCountdown().then(() => startRecording());
      } else {
        stopRecording();
      }
    } else {
      // 드웰 중 — 버퍼에 넣지 않음
      const sec = ((TRIGGER_DWELL - dwellCount) / 15).toFixed(1);
      setStatus(isRecording ? `종료 대기 — ${sec}초 유지` : `시작 대기 — ${sec}초 유지`, "processing");
    }
  } else {
    // 존 밖
    if (dwellCount > 0) {
      dwellCount = 0;
      triggerBar.style.background = "#27272a";
      triggerBtn.classList.remove("hand-near");
    }

    if (isRecording) {
      _pushFrame(results);
      if (pauseFrameCount < PAUSE_FRAMES_CLIENT) setStatus("녹화 중...", "recording");
    } else {
      setStatus("버튼에 손을 올리면 시작됩니다", "ready");
    }
  }
}

function _setHandUI(isActive) {
  if (isActive) {
    handIcon.textContent = "🤟";
    handLabel.textContent = "수화 인식 중...";
    handStatus.classList.add("active");
  } else {
    handIcon.textContent = "✋";
    handLabel.textContent = "버튼에 손을 올리면 시작됩니다";
    handStatus.classList.remove("active");
  }
}

// ── Mode toggle ───────────────────────────────────────────────────────────────
chkAuto.addEventListener("change", () => {
  autoMode = chkAuto.checked;
  handStatus.style.display  = autoMode ? "" : "none";
  triggerBtn.style.display  = autoMode ? "" : "none";

  if (!autoMode) {
    btnRecord.disabled = false;
    btnStop.disabled   = true;
    setStatus("수동 모드 — Record 버튼을 눌러 시작", "ready");
  } else {
    btnRecord.disabled = true;
    btnStop.disabled   = true;
    setStatus("자동 모드 — 버튼에 손을 올리면 시작됩니다", "ready");
  }
});

// ── Manual button handlers ────────────────────────────────────────────────────
btnRecord.addEventListener("click", () => {
  if (!autoMode) startRecording();
});
btnStop.addEventListener("click", () => {
  if (!autoMode) stopRecording();
});

// ── Init ─────────────────────────────────────────────────────────────────────
(async function init() {
  try {
    const videoEl  = document.getElementById("camera");
    const canvasEl = document.getElementById("overlay");

    const configPromise = fetchServerConfig();

    setStatus("MediaPipe 로딩 중...");
    await initMediaPipe(videoEl, canvasEl);
    setOnResults(onMediaPipeResults);

    await configPromise;

    btnRecord.disabled = true;
    btnStop.disabled   = true;
    setStatus("자동 모드 — 버튼에 손을 올리면 시작됩니다", "ready");
  } catch (e) {
    setStatus(`초기화 오류: ${e.message}`, "error");
    console.error(e);
  }
})();
