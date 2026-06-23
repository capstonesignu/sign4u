/**
 * recognize_test.js — 인식 테스트 탭
 *
 * DB 녹화 탭과 동일한 트리거 메커니즘:
 *   화면 우측 하단 박스에 손을 1초간 올리면 녹화 시작/종료.
 *   종료 시 /api/recognize 호출 → Top-K 결과 표시.
 */

// ── 상수 (db_record.js / app.js와 동일 값 사용) ───────────────────────────────
const RT_DWELL    = DB_DWELL;     // 8 frames (~0.5 s @ MediaPipe fps)
const RT_COOLDOWN = DB_COOLDOWN;  // 20 frames (트리거 후 무시)
const RT_MIN_FRAMES = 8;

// ── 상태 ──────────────────────────────────────────────────────────────────────
let _rtRecording  = false;
let _rtBuffer     = [];
let _rtDwellCount = 0;
let _rtCooldown   = 0;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const _rtClearBtn  = document.getElementById("btn-rt-clear");
const _rtStatusEl  = document.getElementById("rt-status");
const _rtFrameEl   = document.getElementById("rt-frame-count");
const _rtResultsEl = document.getElementById("rt-results");
const _rtDbInfoEl  = document.getElementById("rt-db-info");
const _rtRecInd    = document.getElementById("rec-indicator");

// ── DB 정보 로드 ───────────────────────────────────────────────────────────────
(async function _loadDbInfo() {
  try {
    const resp = await fetch("/api/config");
    if (!resp.ok) return;
    const cfg = await resp.json();
    if (_rtDbInfoEl) {
      _rtDbInfoEl.textContent =
        cfg.num_vectors > 0
          ? `DB: ${cfg.num_words}개 단어 · ${cfg.num_vectors}개 벡터`
          : "DB 미로드";
    }
  } catch (_) {}
})();

// ── 트리거 존 감지 (db_record.js와 동일) ─────────────────────────────────────
function _rtIsHandInZone(results) {
  const check = (lms) =>
    lms && lms.some(lm => lm.y > TRIGGER_ZONE_Y && lm.x < TRIGGER_ZONE_X);
  return check(results.leftHandLandmarks) || check(results.rightHandLandmarks);
}

// ── 트리거 처리 (db_record.js와 동일한 UX) ────────────────────────────────────
function _rtHandleTrigger(results) {
  if (_rtCooldown > 0) { _rtCooldown--; return; }

  const inZone = _rtIsHandInZone(results);

  if (inZone) {
    _rtDwellCount++;
    const pct      = Math.min(100, (_rtDwellCount / RT_DWELL) * 100);
    const barColor = _rtRecording ? "#ef4444" : "#3b82f6";
    _triggerBar.style.background =
      `linear-gradient(to bottom, ${barColor} ${pct}%, #27272a ${pct}%)`;
    _triggerBtn.classList.add("hand-near");

    if (_rtDwellCount >= RT_DWELL) {
      _rtDwellCount = 0;
      _rtCooldown   = RT_COOLDOWN;
      _triggerBar.style.background = "#27272a";
      _triggerBtn.classList.remove("hand-near");

      if (!_rtRecording) {
        _rtStartRecording();
      } else {
        _rtStopAndRecognize();
      }
    }
  } else {
    if (_rtDwellCount > 0) {
      _rtDwellCount = 0;
      _triggerBar.style.background = "#27272a";
      _triggerBtn.classList.remove("hand-near");
    }
  }
}

// ── 녹화 시작 ─────────────────────────────────────────────────────────────────
function _rtStartRecording() {
  _rtBuffer    = [];
  _rtRecording = true;
  _triggerLabel.textContent = "■";
  _triggerBtn.classList.add("recording-active");
  if (_rtRecInd)  _rtRecInd.classList.remove("hidden");
  if (_rtFrameEl) _rtFrameEl.textContent = "0 frames";
  if (_rtStatusEl) _rtStatusEl.textContent = "녹화 중... (박스에 손을 올리면 종료)";
  if (_rtResultsEl) _rtResultsEl.innerHTML = "";
}

// ── 녹화 종료 + 인식 ──────────────────────────────────────────────────────────
function _rtStopAndRecognize() {
  _rtRecording = false;
  _triggerLabel.textContent = "▶";
  _triggerBtn.classList.remove("recording-active");
  if (_rtRecInd) _rtRecInd.classList.add("hidden");
  _rtDoRecognize();
}

// ── MediaPipe 훅 ──────────────────────────────────────────────────────────────
addOnResults(function _rtHook(results) {
  if (window.currentTab !== "test") return;

  _rtHandleTrigger(results);

  if (!_rtRecording) return;

  const kp   = extractKeypoints(results);
  const flat = flattenKeypoints(kp);
  _rtBuffer.push(flat);
  if (_rtFrameEl) _rtFrameEl.textContent = `${_rtBuffer.length} frames`;
});

// ── 초기화 버튼 ───────────────────────────────────────────────────────────────
if (_rtClearBtn) {
  _rtClearBtn.addEventListener("click", () => {
    _rtRecording  = false;
    _rtBuffer     = [];
    _rtDwellCount = 0;
    _rtCooldown   = 0;
    _triggerLabel.textContent = "▶";
    _triggerBtn.classList.remove("recording-active", "hand-near");
    _triggerBar.style.background = "#27272a";
    if (_rtRecInd)    _rtRecInd.classList.add("hidden");
    if (_rtStatusEl)  _rtStatusEl.textContent = "우측 하단 버튼에 손을 1초간 올리면 시작됩니다";
    if (_rtFrameEl)   _rtFrameEl.textContent  = "0 frames";
    if (_rtResultsEl) _rtResultsEl.innerHTML  = "";
  });
}

// ── 인식 요청 ─────────────────────────────────────────────────────────────────
async function _rtDoRecognize() {
  const n = _rtBuffer.length;
  if (n < RT_MIN_FRAMES) {
    if (_rtStatusEl) _rtStatusEl.textContent =
      `프레임 부족 (${n}/${RT_MIN_FRAMES}). 더 길게 수화하세요.`;
    return;
  }
  if (_rtStatusEl) _rtStatusEl.textContent = `${n} frames 인식 중...`;
  try {
    const resp = await fetch("/api/recognize", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ keypoints: _rtBuffer, word_mode: true }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ error: resp.statusText }));
      throw new Error(err.error || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    _rtRenderResults(data);
    if (_rtStatusEl) {
      const mm = data.margin_mean != null ? data.margin_mean.toFixed(3) : "—";
      _rtStatusEl.textContent = data.accepted
        ? `완료 — ${n} frames · top1=${(data.top1_score ?? 0).toFixed(3)} · margin̄=${mm}`
        : `인식 불가 — 후보가 모호함 (margin̄ ${mm} < ${(data.margin_thr ?? 0).toFixed(2)})`;
    }
  } catch (e) {
    if (_rtStatusEl) _rtStatusEl.textContent = `오류: ${e.message}`;
    console.error("[rt] recognize error", e);
  }
}

// ── 결과 렌더링 ────────────────────────────────────────────────────────────────
// 절대값 바: cosine 0.5→0%, 1.0→100% (score/maxScore 상대정규화 폐기 — top1이 늘 100%로
// 보이던 거짓신뢰 제거). reject 시 결과를 흐리게 + 안내.
const RT_BAR_FLOOR = 0.5;
function _rtRenderResults(data) {
  if (!_rtResultsEl) return;
  const results = (data && data.results) || [];
  if (!results.length) {
    _rtResultsEl.innerHTML = "<div class='rt-empty'>결과 없음</div>";
    return;
  }
  const accepted = data.accepted !== false;
  const margin   = data.margin;
  const marginMean = data.margin_mean;

  let banner = "";
  if (!accepted) {
    const mm = marginMean != null ? marginMean.toFixed(3) : "—";
    banner = `<div class="rt-reject">⚠ 인식 불가 — 후보 간 차이(margin̄ ${mm})가 작아 확신할 수 없습니다</div>`;
  }

  const rows = results.map((r, i) => {
    const pct  = Math.max(0, Math.min(100,
      (r.score - RT_BAR_FLOOR) / (1 - RT_BAR_FLOOR) * 100)).toFixed(1);
    const top1 = (i === 0 && accepted) ? " rt-top1" : "";
    const mTag = (i === 0 && margin != null)
      ? `<span class="rt-margin">Δ${margin.toFixed(3)}</span>` : "";
    return `<div class="rt-row${top1}">
      <span class="rt-rank">${r.rank}</span>
      <span class="rt-word">${r.word}</span>
      ${mTag}
      <div class="rt-bar-wrap"><div class="rt-bar" style="width:${pct}%"></div></div>
      <span class="rt-score">${r.score.toFixed(3)}</span>
    </div>`;
  }).join("");

  _rtResultsEl.innerHTML = banner +
    `<div class="${accepted ? "" : "rt-dim"}">${rows}</div>`;
}

// ── 탭 전환 ───────────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const tab = btn.dataset.tab;

    document.body.dataset.tab = tab;

    const dbPanel   = document.getElementById("db-panel");
    const testPanel = document.getElementById("test-panel");
    if (dbPanel)   dbPanel.classList.toggle("hidden",   tab !== "db");
    if (testPanel) testPanel.classList.toggle("hidden", tab !== "test");

    if (tab !== "test") {
      // 테스트 탭 이탈 시 트리거 상태 초기화
      if (_rtRecording) {
        _rtRecording = false;
        _triggerLabel.textContent = "▶";
        _triggerBtn.classList.remove("recording-active", "hand-near");
        _triggerBar.style.background = "#27272a";
        if (_rtRecInd) _rtRecInd.classList.add("hidden");
      }
      return;
    }

    // 테스트 탭 진입 시 초기화
    _rtBuffer     = [];
    _rtDwellCount = 0;
    _rtCooldown   = 0;
    _rtRecording  = false;
    _triggerLabel.textContent = "▶";
    _triggerBtn.classList.remove("recording-active", "hand-near");
    _triggerBar.style.background = "#27272a";
    if (_rtRecInd)    _rtRecInd.classList.add("hidden");
    if (_rtStatusEl)  _rtStatusEl.textContent = "우측 하단 버튼에 손을 1초간 올리면 시작됩니다";
    if (_rtFrameEl)   _rtFrameEl.textContent  = "0 frames";
    if (_rtResultsEl) _rtResultsEl.innerHTML  = "";
  });
});
