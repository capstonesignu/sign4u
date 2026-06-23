/**
 * Results panel rendering.
 */

function showSentenceResults(data) {
  const sentencePanel = document.getElementById("sentence-panel");
  const wordsEl  = document.getElementById("sentence-words");
  const metaEl   = document.getElementById("sentence-meta");

  document.getElementById("placeholder").classList.add("hidden");
  document.getElementById("top1-panel").classList.add("hidden");
  document.getElementById("top5-panel").classList.add("hidden");
  document.getElementById("top10-panel").classList.add("hidden");

  if (!data.words || data.words.length === 0) {
    wordsEl.innerHTML = "<span class='no-result'>인식된 단어 없음</span>";
    metaEl.textContent = `(${data.n_frames} frames, 세그먼트 없음)`;
    sentencePanel.classList.remove("hidden");
    return;
  }

  wordsEl.innerHTML = "";
  data.words.forEach((topWord, i) => {
    const span = data.spans ? data.spans[i] : null;
    const alts = (data.alternatives && data.alternatives[i]) || [{ word: topWord, score: 0 }];

    const card = document.createElement("div");
    card.className = "result-card";

    // 헤더: 세그먼트 번호 + 프레임 범위
    const header = document.createElement("div");
    header.className = "result-card-header";
    header.textContent = span ? `세그먼트 ${i + 1}  (${span[0]}~${span[1]}f)` : `세그먼트 ${i + 1}`;
    card.appendChild(header);

    // 후보 리스트
    alts.forEach((cand, rank) => {
      const word  = typeof cand === "string" ? cand : cand.word;
      const score = typeof cand === "string" ? 0 : (cand.score || 0);
      const pct   = Math.min(100, Math.max(0, score * 100)).toFixed(1);

      const row = document.createElement("div");
      row.className = "cand-row" + (rank === 0 ? " cand-top1" : "");

      row.innerHTML = `
        <span class="cand-rank">${rank + 1}</span>
        <span class="cand-word">${word}</span>
        <div class="cand-bar-wrap">
          <div class="cand-bar" style="width:${pct}%"></div>
        </div>
        <span class="cand-score">${pct}%</span>
      `;
      card.appendChild(row);
    });

    wordsEl.appendChild(card);
  });

  metaEl.textContent = `${data.words.length}개 세그먼트  |  ${data.n_frames} frames`;
  sentencePanel.classList.remove("hidden");
}

function showResults(results) {
  document.getElementById("placeholder").classList.add("hidden");

  const top1 = document.getElementById("top1-panel");
  const top5 = document.getElementById("top5-panel");
  const top10 = document.getElementById("top10-panel");

  if (results.length === 0) {
    top1.classList.add("hidden");
    top5.classList.add("hidden");
    top10.classList.add("hidden");
    document.getElementById("placeholder").classList.remove("hidden");
    return;
  }

  // Top-1
  const first = results[0];
  document.getElementById("top1-word").textContent = first.word;
  document.getElementById("top1-fill").style.width = `${first.score * 100}%`;
  document.getElementById("top1-score").textContent =
    `${(first.score * 100).toFixed(1)}% | ${first.word_id}`;
  top1.classList.remove("hidden");

  // Top-5
  renderList("top5-list", results.slice(0, 5));
  top5.classList.remove("hidden");

  // Top-10
  if (results.length > 5) {
    renderList("top10-list", results);
    top10.classList.remove("hidden");
  } else {
    top10.classList.add("hidden");
  }
}

function renderList(elId, items) {
  const ol = document.getElementById(elId);
  ol.innerHTML = "";
  for (const item of items) {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="rank-num">${item.rank}</span>
      <span class="result-word">${item.word}<span class="result-id">${item.word_id}</span></span>
      <div class="result-score-bar"><div class="fill" style="width:${item.score * 100}%"></div></div>
      <span class="result-score-text">${(item.score * 100).toFixed(1)}%</span>
    `;
    ol.appendChild(li);
  }
}

function setStatus(text, type) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = "status";
  if (type) el.classList.add(type);
}

function setFrameCount(n) {
  document.getElementById("frame-counter").textContent = `${n} frames`;
}

function setServerInfo(info) {
  const el = document.getElementById("server-info");
  el.textContent =
    `Server: preset=${info.feature_preset}, ` +
    `dim=${info.input_dim}→${info.d_model}, words=${info.num_words}, vectors=${info.num_vectors}`;
}
