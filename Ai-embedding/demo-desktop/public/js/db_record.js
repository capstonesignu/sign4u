/**
 * DB 녹화 탭 — 트리거 버튼(인식 탭과 동일) → 영상+키포인트 미리보기 → 싱크 조정 → 타임라인 crop → DB 저장.
 *
 * 상태머신: idle → recording → captured
 *   idle:     버튼 존에 손 올리면 1초 카운트다운 후 녹화 시작
 *   recording: kp 수집 중, 다시 버튼 존에 손 올리면 1초 후 종료
 *   captured:  영상+키포인트 미리보기, 싱크 조정, 타임라인 crop 도구 표시
 *
 * 싱크 원리:
 *   탭 진입 즉시 MediaRecorder 백그라운드 녹화 시작 → _vidStartTs 기록.
 *   트리거 후 카운트다운 완료 시 _dbVideoOffset = (now - _vidStartTs) / 1000.
 *   kp[i] ↔ video.currentTime = _dbVideoOffset + i / DB_KP_FPS
 *   _syncAdjustFrames로 사용자가 수동 보정 가능 (±1 kp frame 단위).
 */

// ── Constants ─────────────────────────────────────────────────────────────────
const DB_KP_FPS          = 15;   // 저장(다운샘플) 목표 fps; 수집은 MediaPipe 전체 fps로
const DB_MIN_WORD_FRAMES = 10;
const DB_DWELL           = 8;    // trigger 유지 프레임 수 (~0.5 s @ 15 fps MediaPipe)
const DB_COOLDOWN        = 20;   // trigger 후 무시 프레임 수

// 포즈 landmark 연결 (extractKeypoints의 9-landmark 순서 기준)
const POSE_CONNS_COMPACT = [
  [1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8],
];

// ── State ─────────────────────────────────────────────────────────────────────
let _dbState      = "idle";
let _dbBuffer     = [];      // kp frames (MediaPipe 전체 fps)
let _dbBufferTs   = [];      // 각 프레임의 video timestamp (bg 녹화 시작 기준 초)
let _dbCapture    = [];      // 캡처 확정 후 kp frames (전체 fps, 프리뷰 싱크용)
let _dbCaptureTs  = [];      // 캡처 확정 후 timestamps

// Trigger
let _dbDwellCount = 0;
let _dbCooldown   = 0;

// Crop state (frame indices into _dbCapture)
let _cropStart    = 0;
let _cropEnd      = 0;

// Selected word (set by wordlist click)
let _dbSelectedWord  = "";
let _dbSelectedUuid  = "";

// Sync
let _vidStartTs      = 0;    // performance.now() when bg recorder started
let _dbVideoOffset   = 0;    // video.currentTime corresponding to kp[0]
let _syncAdjustFrames = 2;   // user adjustment (±frames); default +2

// Video recording
let _mediaRecorder  = null;
let _videoChunks    = [];
let _videoObjectURL = null;
let _kpAnimId       = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const _dbStatusEl       = document.getElementById("db-capture-status");
const _dbPreviewArea    = document.getElementById("db-preview-area");
const _dbPreviewVideo   = document.getElementById("db-preview-video");
const _dbKpCanvas       = document.getElementById("db-kp-canvas");
const _dbTimelineWrap   = document.getElementById("db-timeline-wrap");
const _dbTimelineOuter  = document.getElementById("db-timeline-outer");
const _dbTimelineCanvas = document.getElementById("db-timeline-canvas");
const _dbCropRegion     = document.getElementById("db-crop-region");
const _dbHandleStart    = document.getElementById("db-handle-start");
const _dbHandleEnd      = document.getElementById("db-handle-end");
const _dbTimelineInfo   = document.getElementById("db-timeline-info");
const _dbActionRowEl    = document.getElementById("db-action-row");
const _dbSyncRowEl      = document.getElementById("db-sync-row");
const _dbSyncValEl      = document.getElementById("db-sync-val");
const _dbSaveFormEl     = document.getElementById("db-save-form");
const _dbWordSearchEl   = document.getElementById("db-word-search");
const _dbSaveWordEl     = document.getElementById("db-save-word-display");
const _dbSaveBtn        = document.getElementById("btn-db-save");
const _dbRetakeBtn      = document.getElementById("btn-db-retake");

const _triggerBtn   = document.getElementById("trigger-btn");
const _triggerBar   = document.getElementById("trigger-bar");
const _triggerLabel = document.getElementById("trigger-label");

// ── Tab switching ─────────────────────────────────────────────────────────────
window.currentTab = "db";

document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;
    if (tab === window.currentTab) return;
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    window.currentTab = tab;
  });
});

// ── MediaPipe hook ────────────────────────────────────────────────────────────
let _bgStarted = false;
window._dbRecordingHook = function (results) {
  if (window.currentTab !== "db") return;

  // Lazy-start bg recording on first camera frame
  if (!_bgStarted && _dbState === "idle") {
    _bgStarted = true;
    _startBgRecording();
  }

  if (_dbState === "captured") return;

  // 트리거 감지 (MediaPipe 전체 fps로 실행, 쓰로틀 없음)
  _dbHandleTrigger(results);

  // kp 수집 (MediaPipe 전체 fps, 쓰로틀 없음)
  if (_dbState !== "recording") return;
  const now     = performance.now();
  const videoTs = _vidStartTs > 0 ? (now - _vidStartTs) / 1000 : _dbBuffer.length / DB_KP_FPS;
  _dbBuffer.push(extractKeypoints(results));
  _dbBufferTs.push(videoTs);
  const _elapsed = _dbBufferTs.length > 1 ? _dbBufferTs[_dbBufferTs.length - 1] - _dbBufferTs[0] : 0;
  const _liveFps = _elapsed > 0 ? (_dbBuffer.length - 1) / _elapsed : 0;
  _dbStatusEl.textContent = `녹화 중... ${_dbBuffer.length} frames | ${_elapsed.toFixed(1)}s | ${_liveFps.toFixed(1)} fps`;
  const _fc = document.getElementById("frame-counter");
  if (_fc) _fc.textContent = `${_dbBuffer.length}f  ${_elapsed.toFixed(1)}s  ${_liveFps.toFixed(1)}fps`;
};

// ── Trigger zone detection ────────────────────────────────────────────────────
function _dbIsHandInZone(results) {
  const check = (lms) => lms && lms.some(lm => lm.y > TRIGGER_ZONE_Y && lm.x < TRIGGER_ZONE_X);
  return check(results.leftHandLandmarks) || check(results.rightHandLandmarks);
}

function _dbHandleTrigger(results) {
  if (_dbCooldown > 0) { _dbCooldown--; return; }

  const inZone = _dbIsHandInZone(results);

  if (inZone) {
    _dbDwellCount++;
    const pct      = Math.min(100, (_dbDwellCount / DB_DWELL) * 100);
    const barColor = _dbState === "recording" ? "#ef4444" : "#3b82f6";
    _triggerBar.style.background =
      `linear-gradient(to bottom, ${barColor} ${pct}%, #27272a ${pct}%)`;
    _triggerBtn.classList.add("hand-near");

    if (_dbDwellCount >= DB_DWELL) {
      _dbDwellCount = 0;
      _dbCooldown   = DB_COOLDOWN;
      _triggerBar.style.background = "#27272a";
      _triggerBtn.classList.remove("hand-near");

      if (_dbState === "idle") {
        _showCountdown().then(() => {
          _dbVideoOffset = _vidStartTs > 0 ? (performance.now() - _vidStartTs) / 1000 : 0;
          _dbBuffer      = [];
          _dbLastTs      = 0;
          _dbState       = "recording";
          _triggerLabel.textContent = "■";
          _triggerBtn.classList.add("recording-active");
          _dbStatusEl.textContent = "녹화 중... 0 frames";
          _dbStatusEl.className   = "db-capture-status recording";
        });
      } else if (_dbState === "recording") {
        _showCountdown().then(() => {
          _dbCapture   = [..._dbBuffer];
          _dbCaptureTs = [..._dbBufferTs];
          _triggerLabel.textContent = "▶";
          _triggerBtn.classList.remove("recording-active");
          if (_dbCapture.length >= DB_MIN_WORD_FRAMES) {
            _dbState = "captured";
            _stopBgRecordingAndBuild().then(url => _showCaptureUI(url));
          } else {
            _dbReset();
          }
        });
      }
    }
  } else {
    if (_dbDwellCount > 0) {
      _dbDwellCount = 0;
      _triggerBar.style.background = "#27272a";
      _triggerBtn.classList.remove("hand-near");
    }
    if (_dbState === "idle") {
      _dbStatusEl.textContent = "버튼에 손을 올리면 시작됩니다";
      _dbStatusEl.className   = "db-capture-status";
    }
  }
}

// ── Background video recording ────────────────────────────────────────────────
function _startBgRecording() {
  const stream = document.getElementById("camera")?.srcObject;
  if (!stream || !window.MediaRecorder) return;

  _stopBgRecording();

  const mimeType = ["video/webm;codecs=vp8", "video/webm", "video/mp4"]
    .find(t => MediaRecorder.isTypeSupported(t)) || "";
  try {
    _videoChunks   = [];
    _mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
    _mediaRecorder.ondataavailable = e => { if (e.data.size > 0) _videoChunks.push(e.data); };
    _mediaRecorder.start(100);
    _vidStartTs = performance.now();
  } catch (e) {
    _mediaRecorder = null;
    _vidStartTs    = 0;
  }
}

function _stopBgRecording() {
  if (_mediaRecorder && _mediaRecorder.state !== "inactive") {
    try { _mediaRecorder.stop(); } catch (_) {}
  }
  _mediaRecorder = null;
  _videoChunks   = [];
}

function _stopBgRecordingAndBuild() {
  return new Promise(resolve => {
    if (!_mediaRecorder || _mediaRecorder.state === "inactive") {
      resolve(null);
      return;
    }
    _mediaRecorder.onstop = () => {
      const mimeType = _videoChunks[0]?.type || "video/webm";
      const blob = new Blob(_videoChunks, { type: mimeType });
      if (_videoObjectURL) URL.revokeObjectURL(_videoObjectURL);
      _videoObjectURL = URL.createObjectURL(blob);
      resolve(_videoObjectURL);
    };
    _mediaRecorder.stop();
  });
}

// ── Sync helpers ──────────────────────────────────────────────────────────────
// 실제 수집 타임스탬프(_dbCaptureTs)를 사용해 video time ↔ kp frame 변환.
// 고정 fps 가정을 없애므로 실제 수집 fps와 무관하게 정확한 싱크 보장.
// _syncAdjustFrames: 사용자 보정 (− = kp 앞당김, + = kp 늦춤)

function _kpFrameToVideoTime(frame) {
  if (_dbCaptureTs.length === 0) return _dbVideoOffset + frame / DB_KP_FPS;
  const f = Math.max(0, Math.min(frame, _dbCaptureTs.length - 1));
  // 수집 ts에서 sync 보정량(초)을 빼면 video에서 그 kp를 보여줄 시각
  return _dbCaptureTs[f] - _syncAdjustFrames / DB_KP_FPS;
}

function _videoTimeToKpFrame(videoTime) {
  const ts = _dbCaptureTs;
  if (ts.length === 0) return 0;
  // sync 보정: videoTime에 _syncAdjustFrames/fps만큼 더한 시각에 해당하는 kp 탐색
  const t = videoTime + _syncAdjustFrames / DB_KP_FPS;
  if (t <= ts[0]) return 0;
  if (t >= ts[ts.length - 1]) return ts.length - 1;
  // binary search for closest timestamp
  let lo = 0, hi = ts.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] <= t) lo = mid; else hi = mid;
  }
  return Math.abs(ts[lo] - t) <= Math.abs(ts[hi] - t) ? lo : hi;
}

// ── Sync adjustment buttons ───────────────────────────────────────────────────
document.getElementById("btn-sync-minus").addEventListener("click", () => {
  _syncAdjustFrames--;
  _dbSyncValEl.textContent = `${_syncAdjustFrames > 0 ? "+" : ""}${_syncAdjustFrames} f`;
});

document.getElementById("btn-sync-plus").addEventListener("click", () => {
  _syncAdjustFrames++;
  _dbSyncValEl.textContent = `${_syncAdjustFrames > 0 ? "+" : ""}${_syncAdjustFrames} f`;
});

// ── Captured UI setup ─────────────────────────────────────────────────────────
const DB_TRIM_FRAMES = 5;   // 앞뒤 자동 trim

function _showCaptureUI(videoURL) {
  const trim    = Math.min(DB_TRIM_FRAMES, Math.floor((_dbCapture.length - DB_MIN_WORD_FRAMES) / 2));
  _cropStart    = Math.max(0, trim);
  _cropEnd      = Math.min(_dbCapture.length - 1, _dbCapture.length - 1 - trim);
  _syncAdjustFrames = 2;
  _dbSyncValEl.textContent = "+2 f";

  const _capDur = _dbCaptureTs.length > 1 ? _dbCaptureTs[_dbCaptureTs.length - 1] - _dbCaptureTs[0] : 0;
  const _capFps = _capDur > 0 ? (_dbCapture.length - 1) / _capDur : 0;
  _dbStatusEl.textContent = `${_dbCapture.length} frames | ${_capDur.toFixed(1)}s | ${_capFps.toFixed(1)} fps — 싱크 조정 후 타임라인으로 구간을 조절하세요`;
  _dbStatusEl.className   = "db-capture-status captured";

  if (videoURL) {
    _dbPreviewVideo.src = videoURL;
    _dbPreviewVideo.load();
    _dbPreviewVideo.addEventListener("loadeddata", () => {
      _dbPreviewVideo.currentTime = _kpFrameToVideoTime(0);
      _dbPreviewVideo.play().catch(() => {});
      _startKpOverlay();
    }, { once: true });
    _dbPreviewArea.classList.remove("hidden");
  }

  _dbTimelineWrap.classList.remove("hidden");
  requestAnimationFrame(() => {
    _drawTimeline();
    _positionHandles();
    _updateTimelineInfo();
  });

  _dbActionRowEl.classList.remove("hidden");
  _dbSyncRowEl.classList.remove("hidden");
  _dbSaveFormEl.classList.remove("hidden");
  _dbSaveBtn.disabled   = false;
  _dbRetakeBtn.disabled = false;
  if (_dbWordSearchEl) _dbWordSearchEl.focus();

  _dbPreviewVideo.addEventListener("timeupdate", _onVideoTimeUpdate);
}

function _onVideoTimeUpdate() {
  const endTime = _kpFrameToVideoTime(_cropEnd + 1);
  if (_dbPreviewVideo.currentTime > endTime + 0.1) {
    _dbPreviewVideo.currentTime = Math.max(0, _kpFrameToVideoTime(_cropStart));
  }
}

// ── Keypoint overlay ──────────────────────────────────────────────────────────
function _startKpOverlay() {
  if (_kpAnimId) cancelAnimationFrame(_kpAnimId);

  function loop() {
    if (_dbState !== "captured") return;
    const frameIdx = _videoTimeToKpFrame(_dbPreviewVideo.currentTime);
    _renderKpCanvas(frameIdx);
    _kpAnimId = requestAnimationFrame(loop);
  }
  _kpAnimId = requestAnimationFrame(loop);
}

let _kpRenderCount = 0;

function _renderKpCanvas(frameIdx) {
  const video  = _dbPreviewVideo;
  const canvas = _dbKpCanvas;
  const ctx    = canvas.getContext("2d");

  const W = Math.round(canvas.offsetWidth  || video.offsetWidth  || 320);
  const H = Math.round(canvas.offsetHeight || video.offsetHeight || 240);
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width  = W;
    canvas.height = H;
  }

  if (_kpRenderCount < 5) {
    const frame = _dbCapture[frameIdx];
    console.log("[kp-overlay] render", _kpRenderCount, {
      W, H, frameIdx,
      captureLen: _dbCapture.length,
      hasPose: !!frame?.pose,
      hasLeftHand: !!frame?.leftHand,
      hasRightHand: !!frame?.rightHand,
      hasFace: !!frame?.face,
    });
    _kpRenderCount++;
  }

  ctx.clearRect(0, 0, W, H);

  const frame = _dbCapture[frameIdx];
  if (!frame) return;

  const vW    = video.videoWidth  || W;
  const vH    = video.videoHeight || H;
  const scale = Math.max(W / vW, H / vH);
  const offX  = (vW * scale - W) / 2;
  const offY  = (vH * scale - H) / 2;

  const toLm = (arr, i) => arr ? {
    x: arr[i * 3]     * vW * scale - offX,
    y: arr[i * 3 + 1] * vH * scale - offY,
    ok: arr[i * 3] !== 0 || arr[i * 3 + 1] !== 0,
  } : null;

  // Pose skeleton
  if (frame.pose) {
    ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 2;
    for (const [a, b] of POSE_CONNS_COMPACT) {
      const pa = toLm(frame.pose, a), pb = toLm(frame.pose, b);
      if (!pa?.ok || !pb?.ok) continue;
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    }
    ctx.fillStyle = "#4ade80";
    for (let i = 0; i < 9; i++) {
      const p = toLm(frame.pose, i);
      if (!p?.ok) continue;
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI); ctx.fill();
    }
  }

  // Left hand
  if (frame.leftHand) {
    ctx.fillStyle = "#f87171";
    for (let i = 0; i < 21; i++) {
      const p = toLm(frame.leftHand, i);
      if (!p?.ok) continue;
      ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI); ctx.fill();
    }
  }

  // Right hand
  if (frame.rightHand) {
    ctx.fillStyle = "#60a5fa";
    for (let i = 0; i < 21; i++) {
      const p = toLm(frame.rightHand, i);
      if (!p?.ok) continue;
      ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, 2 * Math.PI); ctx.fill();
    }
  }

  // Face
  if (frame.face) {
    ctx.fillStyle = "#facc15";
    for (let i = 0; i < 19; i++) {
      const p = toLm(frame.face, i);
      if (!p?.ok) continue;
      ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI); ctx.fill();
    }
  }
}

// ── Timeline ──────────────────────────────────────────────────────────────────
function _dbVel(a, b) {
  let sum = 0;
  if (a.leftHand  && b.leftHand)  sum += (a.leftHand[0]  - b.leftHand[0])  ** 2 + (a.leftHand[1]  - b.leftHand[1])  ** 2;
  if (a.rightHand && b.rightHand) sum += (a.rightHand[0] - b.rightHand[0]) ** 2 + (a.rightHand[1] - b.rightHand[1]) ** 2;
  return Math.sqrt(sum);
}

function _drawTimeline() {
  const canvas = _dbTimelineCanvas;
  const W = _dbTimelineOuter.clientWidth || 400;
  const H = canvas.height;
  canvas.width = W;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);

  const n = _dbCapture.length;
  if (n < 2) return;

  const vels = [0];
  for (let i = 1; i < n; i++) vels.push(_dbVel(_dbCapture[i], _dbCapture[i - 1]));
  const maxVel = Math.max(...vels, 0.001);
  const barW   = W / n;

  ctx.fillStyle = "#3f3f46";
  for (let i = 0; i < n; i++) {
    const h = Math.max(2, (vels[i] / maxVel) * H);
    ctx.fillRect(i * barW, H - h, Math.max(1, barW - 0.5), h);
  }

  ctx.fillStyle = "#22c55e";
  for (let i = _cropStart; i <= _cropEnd; i++) {
    const h = Math.max(2, (vels[i] / maxVel) * H);
    ctx.fillRect(i * barW, H - h, Math.max(1, barW - 0.5), h);
  }
}

function _positionHandles() {
  const W = _dbTimelineOuter.clientWidth || 400;
  const n = _dbCapture.length || 1;
  const xStart = (_cropStart / n) * W;
  const xEnd   = ((_cropEnd + 1) / n) * W;
  _dbHandleStart.style.left = `${xStart}px`;
  _dbHandleEnd.style.left   = `${xEnd - 20}px`;
  _dbCropRegion.style.left  = `${xStart}px`;
  _dbCropRegion.style.width = `${xEnd - xStart}px`;
}

function _updateTimelineInfo() {
  const count = _cropEnd - _cropStart + 1;
  _dbTimelineInfo.textContent = `${_cropStart}~${_cropEnd} frames (${count}개 선택)`;
}

// ── Drag handles ──────────────────────────────────────────────────────────────
let _dragging = null;

_dbHandleStart.addEventListener("mousedown",  e => { e.preventDefault(); _dragging = "start"; });
_dbHandleEnd.addEventListener("mousedown",    e => { e.preventDefault(); _dragging = "end"; });
_dbHandleStart.addEventListener("touchstart", e => { e.preventDefault(); _dragging = "start"; }, { passive: false });
_dbHandleEnd.addEventListener("touchstart",   e => { e.preventDefault(); _dragging = "end"; },   { passive: false });
document.addEventListener("mouseup",  () => { _dragging = null; });
document.addEventListener("touchend", () => { _dragging = null; });

document.addEventListener("mousemove",  _onDragMove);
document.addEventListener("touchmove",  _onDragMove, { passive: false });

function _onDragMove(e) {
  if (!_dragging || _dbState !== "captured") return;
  e.preventDefault();
  const rect  = _dbTimelineOuter.getBoundingClientRect();
  const x     = Math.max(0, Math.min((e.touches ? e.touches[0].clientX : e.clientX) - rect.left, rect.width));
  const n     = _dbCapture.length;
  const frame = Math.round((x / rect.width) * n);

  if (_dragging === "start") {
    _cropStart = Math.max(0, Math.min(frame, _cropEnd - 1));
    _dbPreviewVideo.currentTime = Math.max(0, _kpFrameToVideoTime(_cropStart));
  } else {
    _cropEnd = Math.max(_cropStart + 1, Math.min(frame, n - 1));
  }
  _drawTimeline();
  _positionHandles();
  _updateTimelineInfo();
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function _dbReset() {
  _dbState          = "idle";
  _dbBuffer         = [];
  _dbBufferTs       = [];
  _dbCapture        = [];
  _dbCaptureTs      = [];
  _dbDwellCount     = 0;
  _dbCooldown       = 0;
  _cropStart        = 0;
  _cropEnd          = 0;
  _dbVideoOffset    = 0;
  _syncAdjustFrames = 2;
  _dbSyncValEl.textContent = "+2 f";

  if (_kpAnimId) { cancelAnimationFrame(_kpAnimId); _kpAnimId = null; }

  _dbPreviewVideo.removeEventListener("timeupdate", _onVideoTimeUpdate);
  _dbPreviewVideo.pause();
  _dbPreviewVideo.src = "";
  if (_videoObjectURL) { URL.revokeObjectURL(_videoObjectURL); _videoObjectURL = null; }

  const ctx = _dbKpCanvas.getContext("2d");
  ctx.clearRect(0, 0, _dbKpCanvas.width, _dbKpCanvas.height);
  const tctx = _dbTimelineCanvas.getContext("2d");
  tctx.clearRect(0, 0, _dbTimelineCanvas.width, _dbTimelineCanvas.height);

  _dbPreviewArea.classList.add("hidden");
  _dbTimelineWrap.classList.add("hidden");
  _dbActionRowEl.classList.add("hidden");
  _dbSyncRowEl.classList.add("hidden");
  _dbSaveFormEl.classList.add("hidden");
  // 선택된 단어가 있으면 참고 영상 다시 표시, 없으면 숨김
  if (_dbSelectedUuid && typeof window._dbPlayRefVideo === "function") {
    // reset uuid so it re-loads properly
    const refVideo = document.getElementById("db-ref-video");
    if (refVideo) refVideo.dataset.uuid = "";
    window._dbPlayRefVideo(_dbSelectedUuid, _dbSelectedWord);
  }
  _dbSaveBtn.disabled   = false;
  _dbRetakeBtn.disabled = false;

  _triggerLabel.textContent = "▶";
  _triggerBtn.classList.remove("recording-active", "hand-near");
  _triggerBar.style.background = "#27272a";

  _dbStatusEl.textContent = _dbSelectedWord
    ? `"${_dbSelectedWord}" — 버튼에 손을 올리면 녹화가 시작됩니다`
    : "버튼에 손을 올리면 녹화가 시작됩니다";
  _dbStatusEl.className   = "db-capture-status";

  _bgStarted = true;  // already starting below
  _startBgRecording();
}

// ── fps 다운샘플 (DB 저장 / 다운로드용) ──────────────────────────────────────
function _downsampleToFps(frames, timestamps, targetFps) {
  if (frames.length === 0) return [];
  const out      = [];
  const interval = 1 / targetFps;
  let   nextTs   = timestamps[0];
  for (let i = 0; i < frames.length; i++) {
    if (timestamps[i] >= nextTs) {
      out.push(frames[i]);
      nextTs = timestamps[i] + interval;
    }
  }
  return out;
}

function _downsampleTo15fps(frames, timestamps) {
  return _downsampleToFps(frames, timestamps, DB_KP_FPS);
}

// ── Button handlers ───────────────────────────────────────────────────────────
_dbRetakeBtn.addEventListener("click", () => _dbReset());

document.getElementById("btn-db-replay").addEventListener("click", () => {
  if (_dbState !== "captured" || !_dbPreviewVideo.src) return;
  _dbPreviewVideo.currentTime = Math.max(0, _kpFrameToVideoTime(_cropStart));
  _dbPreviewVideo.play().catch(() => {});
});

_dbSaveBtn.addEventListener("click", async () => {
  const label = _dbSelectedWord.trim();
  if (!label) {
    _dbStatusEl.textContent = "왼쪽 목록에서 단어를 먼저 선택하세요";
    _dbStatusEl.className   = "db-capture-status error";
    return;
  }

  const cropFrames = _dbCapture.slice(_cropStart, _cropEnd + 1);
  const cropTs     = _dbCaptureTs.slice(_cropStart, _cropEnd + 1);

  if (cropFrames.length < 8) {
    _dbStatusEl.textContent = "구간이 너무 짧습니다 (최소 8 frames)";
    _dbStatusEl.className   = "db-capture-status error";
    return;
  }

  const duration  = cropTs[cropTs.length - 1] - cropTs[0];
  const actualFps = duration > 0 ? (cropFrames.length - 1) / duration : 30.0;

  _dbSaveBtn.disabled   = true;
  _dbRetakeBtn.disabled = true;
  _dbStatusEl.textContent = "저장 중...";
  _dbStatusEl.className   = "db-capture-status processing";

  try {
    const recorderName = window._currentUser?.full_name || window._currentUser?.username || "";
    const res = await fetch("/data/keypoints/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keypoints: cropFrames.map(flattenKeypoints), fps: actualFps, label, uuid: _dbSelectedUuid, recorder_name: recorderName }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    _dbStatusEl.textContent =
      `저장 완료: "${label}"  (${cropFrames.length}f @ ${actualFps.toFixed(1)}fps)  →  ${data.saved_npz ? data.saved_npz.split("/").pop() : "저장됨"}`;
    _dbStatusEl.className = "db-capture-status done";
    _fetchRecordingsList();
    setTimeout(() => _dbReset(), 2500);
  } catch (e) {
    _dbStatusEl.textContent = `오류: ${e.message}`;
    _dbStatusEl.className   = "db-capture-status error";
    _dbSaveBtn.disabled   = false;
    _dbRetakeBtn.disabled = false;
  }
});

// ── 참고 영상 로더 ────────────────────────────────────────────────────────────
// uuid: jbedu entry uuid, word: Korean label for display
window._dbPlayRefVideo = function(uuid, word) {
  const refVideoRow = document.getElementById("db-ref-video-row");
  const refVideo    = document.getElementById("db-ref-video");
  if (!refVideoRow || !refVideo || !uuid) {
    if (refVideoRow) refVideoRow.classList.add("hidden");
    return;
  }
  if (refVideo.dataset.uuid === uuid) return;  // already loaded
  refVideo.dataset.uuid = uuid;
  refVideo.src = `/data/video/${uuid}`;
  refVideo.load();
  refVideo.play().catch(() => {});
  refVideoRow.classList.remove("hidden");
};

window._dbHideRefVideo = function() {
  const refVideoRow = document.getElementById("db-ref-video-row");
  const refVideo    = document.getElementById("db-ref-video");
  if (!refVideo || !refVideoRow) return;
  refVideo.pause();
  refVideo.dataset.uuid = "";
  refVideoRow.classList.add("hidden");
};

// ── DB 단어 목록 ──────────────────────────────────────────────────────────────
let _jbeduEntries          = [];
let _recordingCounts       = {};   // korean_word → count (녹화 목록 패널용)
let _recordingCountsByUuid = {};   // jbedu uuid  → count (단어 목록 badge용)

function _makeWordItem(entry, listEl) {
  const cnt  = _recordingCountsByUuid[entry.uuid] || 0;
  const item = document.createElement("div");
  item.className = "db-wordlist-item";
  item.dataset.uuid = entry.uuid;
  item.dataset.word = entry.korean_word;
  item.innerHTML = `<span class="db-wordlist-name">${entry.korean_word}</span>`
    + `<span class="db-rec-badge${cnt > 0 ? "" : " hidden"}">${cnt}</span>`
    + `<button class="btn-kp-download" title="키포인트 다운로드" data-word="${entry.korean_word}">⬇</button>`;
  item.querySelector(".btn-kp-download").addEventListener("click", (e) => {
    e.stopPropagation();
    const a = document.createElement("a");
    a.href = `/data/keypoints/download/${encodeURIComponent(entry.korean_word)}`;
    a.download = `${entry.korean_word}.zip`;
    a.click();
  });
  item.addEventListener("click", () => {
    listEl.querySelectorAll(".db-wordlist-item").forEach(el => el.classList.remove("active"));
    item.classList.add("active");
    _dbSelectedWord = entry.korean_word;
    _dbSelectedUuid = entry.uuid;
    // 선택된 단어의 카테고리·섹션을 펼침
    const catBody = item.closest(".db-category-body");
    if (catBody) {
      catBody.classList.remove("collapsed");
      const catHdr = catBody.previousElementSibling;
      if (catHdr) catHdr.classList.remove("collapsed");
    }
    const secBody = item.closest(".db-section-body");
    if (secBody) {
      secBody.classList.remove("collapsed");
      const secHdr = secBody.previousElementSibling;
      if (secHdr) secHdr.classList.remove("collapsed");
    }
    if (_dbSaveWordEl) _dbSaveWordEl.textContent = `"${entry.korean_word}"`;
    if (_dbState === "idle") {
      _dbStatusEl.textContent = `"${entry.korean_word}" — 버튼에 손을 올리면 녹화가 시작됩니다`;
    }
    window._dbPlayRefVideo(entry.uuid, entry.korean_word);
  });
  return item;
}

function _dbRenderWordList(entries, flat = false) {
  const listEl = document.getElementById("db-wordlist");
  if (!listEl) return;
  if (entries.length === 0) {
    listEl.innerHTML = '<div class="db-wordlist-empty">검색 결과 없음</div>';
    return;
  }
  listEl.innerHTML = "";

  // 검색 중이면 flat 리스트로 표시
  if (flat) {
    entries.forEach(entry => listEl.appendChild(_makeWordItem(entry, listEl)));
    return;
  }

  // section → category → entries 위계 트리
  const tree = {};
  const SEC_ORDER = ["단어", "문장", "지명", "회화수어"];
  entries.forEach(entry => {
    const sec = entry.section || "기타";
    const cat = entry.category || "기타";
    if (!tree[sec]) tree[sec] = {};
    if (!tree[sec][cat]) tree[sec][cat] = [];
    tree[sec][cat].push(entry);
  });

  const sections = [...SEC_ORDER.filter(s => tree[s]), ...Object.keys(tree).filter(s => !SEC_ORDER.includes(s))];

  sections.forEach(section => {
    const categories = tree[section];

    // 섹션 헤더
    const secHdr = document.createElement("div");
    secHdr.className = "db-section-header";
    const totalCount = Object.values(categories).reduce((s, a) => s + a.length, 0);
    secHdr.innerHTML = `<span class="db-section-toggle">▼</span><span>${section}</span><span class="db-category-count">${totalCount}</span>`;

    const secBody = document.createElement("div");
    secBody.className = "db-section-body collapsed";
    secHdr.classList.add("collapsed");

    secHdr.addEventListener("click", () => {
      const c = secBody.classList.toggle("collapsed");
      secHdr.classList.toggle("collapsed", c);
    });

    // 카테고리
    const catNames = Object.keys(categories).sort((a, b) => a.localeCompare(b, "ko"));
    catNames.forEach(category => {
      const items = categories[category];

      const catHdr = document.createElement("div");
      catHdr.className = "db-category-header";
      catHdr.innerHTML = `<span class="db-category-toggle">▼</span><span>${category}</span><span class="db-category-count">${items.length}</span>`;

      const catBody = document.createElement("div");
      catBody.className = "db-category-body collapsed";
      catHdr.classList.add("collapsed");

      catHdr.addEventListener("click", () => {
        const c = catBody.classList.toggle("collapsed");
        catHdr.classList.toggle("collapsed", c);
      });

      items.forEach(entry => catBody.appendChild(_makeWordItem(entry, listEl)));

      secBody.appendChild(catHdr);
      secBody.appendChild(catBody);
    });

    listEl.appendChild(secHdr);
    listEl.appendChild(secBody);
  });
}

// 검색 필터
if (_dbWordSearchEl) {
  _dbWordSearchEl.addEventListener("input", () => {
    const q = _dbWordSearchEl.value.trim().toLowerCase();
    if (!q) { _dbRenderWordList(_jbeduEntries); return; }
    _dbRenderWordList(_jbeduEntries.filter(e =>
      e.korean_word.toLowerCase().includes(q) ||
      (e.english_word || "").toLowerCase().includes(q) ||
      (e.category || "").includes(q)
    ), true);
  });
}

async function _dbFetchWordList() {
  const listEl = document.getElementById("db-wordlist");
  if (!listEl) return;
  listEl.innerHTML = '<div class="db-wordlist-empty">로딩 중...</div>';
  try {
    let allEntries = [];
    let offset = 0;
    const limit = 1000;
    while (true) {
      const res = await fetch(`/data/entries?has_video=true&limit=${limit}&offset=${offset}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      allEntries = allEntries.concat(data.entries);
      if (allEntries.length >= data.total || data.entries.length < limit) break;
      offset += limit;
    }
    _jbeduEntries = allEntries;
    _dbRenderWordList(_jbeduEntries);
  } catch (e) {
    listEl.innerHTML = `<div class="db-wordlist-empty">로드 실패: ${e.message}</div>`;
  }
}

// 전체 다운로드 버튼
const _btnDownloadAll = document.getElementById("btn-download-all");
if (_btnDownloadAll) {
  _btnDownloadAll.addEventListener("click", () => {
    const a = document.createElement("a");
    a.href = "/data/keypoints/download";
    a.download = "recordings.zip";
    a.click();
  });
}

// ── 녹화된 키포인트 목록 + 삭제 ─────────────────────────────────────────────

function _updateRecordingBadgesInWordlist() {
  const wordlistEl = document.getElementById("db-wordlist");
  if (!wordlistEl) return;
  wordlistEl.querySelectorAll(".db-wordlist-item[data-uuid]").forEach(item => {
    const cnt = _recordingCountsByUuid[item.dataset.uuid] || 0;
    let badge = item.querySelector(".db-rec-badge");
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "db-rec-badge";
      const dl = item.querySelector(".btn-kp-download");
      if (dl) item.insertBefore(badge, dl); else item.appendChild(badge);
    }
    badge.textContent = cnt;
    badge.classList.toggle("hidden", cnt === 0);
  });
  // 탭 버튼 총계 badge 갱신
  const total = Object.values(_recordingCounts).reduce((s, c) => s + c, 0);
  const tabBtn = document.querySelector(".tab-btn[data-tab='db']");
  if (tabBtn) {
    let badge = tabBtn.querySelector(".tab-count");
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "tab-count";
      tabBtn.appendChild(badge);
    }
    badge.textContent = total > 0 ? total : "";
    badge.classList.toggle("hidden", total === 0);
  }
}

async function _fetchRecordingsList() {
  const listEl = document.getElementById("db-recordings-list");
  if (!listEl) return;

  // 현재 펼쳐진 항목 레이블 기억
  const expandedLabels = new Set();
  listEl.querySelectorAll(".db-rec-row").forEach(row => {
    const fl  = row.querySelector(".db-rec-files");
    const lbl = row.querySelector(".db-rec-label")?.textContent;
    if (lbl && fl && !fl.classList.contains("hidden")) expandedLabels.add(lbl);
  });

  listEl.innerHTML = '<div class="db-wordlist-empty">로딩 중...</div>';
  try {
    const res = await fetch("/data/keypoints/list");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const words = data.words || {};
    const entries = Object.entries(words);

    // _recordingCounts / _recordingCountsByUuid 갱신 + 단어 목록 badge 업데이트
    _recordingCounts = {};
    for (const [label, cnt] of entries) _recordingCounts[label] = cnt;
    _recordingCountsByUuid = data.uuids || {};
    _updateRecordingBadgesInWordlist();

    // 통계 표시
    const totalWords      = entries.length;
    const totalRecordings = entries.reduce((s, [, c]) => s + c, 0);
    const statWords = document.getElementById("stat-words");
    const statRecs  = document.getElementById("stat-recordings");
    if (statWords) statWords.textContent = `단어 ${totalWords}개`;
    if (statRecs)  statRecs.textContent  = `녹화 ${totalRecordings}개`;

    if (entries.length === 0) {
      listEl.innerHTML = '<div class="db-wordlist-empty">녹화된 키포인트 없음</div>';
      return;
    }
    listEl.innerHTML = "";
    for (const [label, count] of entries.sort((a, b) => a[0].localeCompare(b[0], "ko"))) {
      const row = document.createElement("div");
      row.className = "db-rec-row";

      const nameEl = document.createElement("span");
      nameEl.className = "db-rec-label";
      nameEl.textContent = label;

      const countEl = document.createElement("span");
      countEl.className = "db-rec-count";
      countEl.textContent = `${count}개`;

      const expandBtn = document.createElement("button");
      expandBtn.className = "btn btn-sync db-rec-expand";
      expandBtn.textContent = "▼";
      expandBtn.title = "파일 목록";

      const dlBtn = document.createElement("button");
      dlBtn.className = "btn btn-sync db-rec-dl";
      dlBtn.textContent = "⬇";
      dlBtn.title = `${label} 다운로드`;
      dlBtn.addEventListener("click", () => {
        const a = document.createElement("a");
        a.href = `/data/keypoints/download/${encodeURIComponent(label)}`;
        a.download = `${label}.zip`;
        a.click();
      });

      const delAllBtn = document.createElement("button");
      delAllBtn.className = "btn btn-stop db-rec-del-all";
      delAllBtn.textContent = "전체 삭제";

      row.appendChild(nameEl);
      row.appendChild(countEl);
      row.appendChild(expandBtn);
      row.appendChild(dlBtn);
      row.appendChild(delAllBtn);
      listEl.appendChild(row);

      // 파일 목록 (접힘)
      const fileList = document.createElement("div");
      fileList.className = "db-rec-files hidden";
      row.appendChild(fileList);

      expandBtn.addEventListener("click", async () => {
        if (!fileList.classList.contains("hidden")) {
          fileList.classList.add("hidden");
          expandBtn.textContent = "▼";
          return;
        }
        fileList.innerHTML = '<span class="db-wordlist-empty">로딩 중...</span>';
        fileList.classList.remove("hidden");
        expandBtn.textContent = "▲";
        try {
          const r = await fetch(`/data/keypoints/list?label=${encodeURIComponent(label)}`);
          const d = await r.json();
          fileList.innerHTML = "";
          for (const fname of (d.files || [])) {
            const frow = document.createElement("div");
            frow.className = "db-rec-file-row";
            const fnameEl = document.createElement("span");
            fnameEl.className = "db-rec-filename";
            fnameEl.textContent = fname;
            const fdelBtn = document.createElement("button");
            fdelBtn.className = "btn btn-stop db-rec-del-file";
            fdelBtn.textContent = "삭제";
            fdelBtn.addEventListener("click", async () => {
              if (!confirm(`"${fname}" 을 삭제할까요?`)) return;
              const dr = await fetch(`/data/keypoints/${encodeURIComponent(label)}/${encodeURIComponent(fname)}`, { method: "DELETE" });
              if (dr.ok) { frow.remove(); _fetchRecordingsList(); }
              else alert("삭제 실패");
            });
            frow.appendChild(fnameEl);
            frow.appendChild(fdelBtn);
            fileList.appendChild(frow);
          }
        } catch (e) {
          fileList.innerHTML = `<span class="db-wordlist-empty">로드 실패: ${e.message}</span>`;
        }
      });

      delAllBtn.addEventListener("click", async () => {
        if (!confirm(`"${label}" 의 모든 녹화(${count}개)를 삭제할까요?`)) return;
        const dr = await fetch(`/data/keypoints/${encodeURIComponent(label)}`, { method: "DELETE" });
        if (dr.ok) { row.remove(); _fetchRecordingsList(); }
        else alert("삭제 실패");
      });
    }
    // 이전에 펼쳐져 있던 항목 복원
    if (expandedLabels.size > 0) {
      listEl.querySelectorAll(".db-rec-row").forEach(row => {
        const lbl = row.querySelector(".db-rec-label")?.textContent;
        if (expandedLabels.has(lbl)) row.querySelector(".db-rec-expand")?.click();
      });
    }
  } catch (e) {
    listEl.innerHTML = `<div class="db-wordlist-empty">로드 실패: ${e.message}</div>`;
  }
}

document.getElementById("btn-refresh-recordings")?.addEventListener("click", _fetchRecordingsList);

// 페이지 로드 시 단어 목록 자동 로드
_dbFetchWordList();
_fetchRecordingsList();
