/**
 * Keypoint recording feature.
 * Buffers raw (210-dim) keypoints from MediaPipe while recording,
 * then POSTs to /api/record/save and triggers a browser file download.
 *
 * Hook: sets window._recordingHook so app.js can forward each frame.
 */

const REC_FPS = 60.0;

let _recBuffer = [];
let _recActive = false;

const btnRecStart = document.getElementById("btn-rec-start");
const btnRecStop  = document.getElementById("btn-rec-stop");
const recLabelEl  = document.getElementById("record-label");
const recStatusEl = document.getElementById("record-status");

// Called by app.js on every MediaPipe frame
window._recordingHook = function (results) {
  if (!_recActive) return;
  _recBuffer.push(extractKeypoints(results));
  recStatusEl.textContent = `녹화 중... ${_recBuffer.length} frames`;
};

btnRecStart.addEventListener("click", () => {
  if (_recActive) return;
  _recBuffer = [];
  _recActive = true;
  btnRecStart.disabled = true;
  btnRecStop.disabled = false;
  recStatusEl.textContent = "녹화 시작 — 수화를 수행하세요";
  recStatusEl.className = "record-status recording";
});

btnRecStop.addEventListener("click", async () => {
  if (!_recActive) return;
  _recActive = false;
  btnRecStart.disabled = false;
  btnRecStop.disabled = true;

  const frames = _recBuffer;
  _recBuffer = [];

  if (frames.length === 0) {
    recStatusEl.textContent = "녹화된 프레임이 없습니다.";
    recStatusEl.className = "record-status error";
    return;
  }

  const label = recLabelEl.value.trim() || "unknown";
  recStatusEl.textContent = `저장 중... (${frames.length} frames)`;
  recStatusEl.className = "record-status processing";

  try {
    const res = await fetch("/api/record/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keypoints: frames, label, fps: REC_FPS }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const rawName  = res.headers.get("X-Filename") || encodeURIComponent(`${label}.npz`);
    const filename = decodeURIComponent(rawName);
    const nFrames  = res.headers.get("X-Frames")   || String(frames.length);

    const blob = await res.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    recStatusEl.textContent = `저장 완료: ${filename} (${nFrames} frames)`;
    recStatusEl.className = "record-status done";
  } catch (e) {
    recStatusEl.textContent = `오류: ${e.message}`;
    recStatusEl.className = "record-status error";
  }
});
