/**
 * Reference panel: searchable word list + sign language video playback.
 * Shows ALL available words from WORD-video directory.
 */

async function initReference() {
  let allWords = [];
  try {
    const res = await fetch("/ref/words");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    allWords = await res.json();
  } catch (e) {
    console.warn("[reference] Failed to load words:", e.message);
    return;
  }

  if (!allWords.length) return;

  const section   = document.getElementById("ref-section");
  const searchEl  = document.getElementById("ref-word-search");
  const listEl    = document.getElementById("ref-word-chips");
  const noSelect  = document.getElementById("ref-no-select");
  const videoWrap = document.getElementById("ref-video-wrap-inner");
  const player    = document.getElementById("ref-video-player");
  const wordLabel = document.getElementById("ref-word-label");

  function resetPlayer() {
    player.pause();
    player.src = "";
    wordLabel.textContent = "";
    noSelect.classList.remove("hidden");
    videoWrap.classList.add("hidden");
  }

  function playWord(item) {
    wordLabel.textContent = item.word;
    player.src = item.video;
    player.load();
    player.play().catch(() => {});
    noSelect.classList.add("hidden");
    videoWrap.classList.remove("hidden");
  }

  function renderList(words) {
    listEl.innerHTML = "";
    words.forEach((item) => {
      const chip = document.createElement("button");
      chip.className = "word-chip";
      chip.textContent = item.word;
      chip.addEventListener("click", () => {
        listEl.querySelectorAll(".word-chip").forEach((c) =>
          c.classList.remove("active")
        );
        chip.classList.add("active");
        playWord(item);
      });
      listEl.appendChild(chip);
    });
  }

  // Initial render
  renderList(allWords);
  resetPlayer();

  // Search filter
  if (searchEl) {
    searchEl.addEventListener("input", () => {
      const q = searchEl.value.trim();
      const filtered = q
        ? allWords.filter((w) => w.word.includes(q))
        : allWords;
      renderList(filtered);
      resetPlayer();
    });
  }

  // Expose for external use (e.g., clicking a recognition result)
  window._refPlayWord = (word) => {
    const item = allWords.find((w) => w.word === word);
    if (!item) return;
    // highlight chip
    listEl.querySelectorAll(".word-chip").forEach((c) =>
      c.classList.remove("active")
    );
    const chip = [...listEl.querySelectorAll(".word-chip")].find(
      (c) => c.textContent === word
    );
    if (chip) {
      chip.classList.add("active");
      chip.scrollIntoView({ block: "nearest" });
    }
    playWord(item);
  };
}

document.addEventListener("DOMContentLoaded", initReference);
