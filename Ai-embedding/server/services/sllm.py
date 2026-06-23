"""Self-contained sLLM service for the KSL production server.

This module is intentionally standalone — it does NOT import from the AI-sLLM
directory because the server and the AI-sLLM repo are separate deployments.

All required pieces (prompt templates, parser, selector, EXAONE 4.0 patch) are
included inline.

Public API
----------
    service = SLLMService(model_path="/path/to/model.gguf", n_threads=4)
    best, all_three = service.normalize_with_candidates(candidates)

where ``candidates`` is a list[list[dict]] of RAG top-K results:
    [
        [{"word": "배", "score": 0.95}, {"word": "복통", "score": 0.82}, ...],
        [{"word": "아프다", "score": 0.90}, ...],
        ...
    ]
"""

from __future__ import annotations

import re
from typing import Optional


# ── Prompt templates (inline copy — must stay in sync with AI-sLLM manually) ──

_NEGATION_WORDS: set[str] = {"없다", "없음", "안", "못", "못하다", "불가능", "불가", "필요없다"}

TOPK_SYSTEM_INSTRUCTION = (
    "너는 한국수어 번역 시스템의 후처리 sLLM이다.\n"
    "각 위치마다 인식 후보 단어들과 유사도 점수가 주어진다.\n"
    "가장 자연스러운 한국어 문장을 1개 생성하라.\n\n"
    "### 필수 규칙 ###\n"
    "1. 반드시 해라체(-다, -ㄴ다, -는다)만 사용한다.\n"
    "2. 과거 표현(어제, 지난 등)이 없으면 현재형으로 쓴다.\n"
    "3. 각 위치의 후보 단어 중 하나만 선택하고, 후보에 없는 단어는 절대 추가하지 않는다.\n"
    "4. 설명 없이 문장 하나만 출력한다.\n"
    "5. 안/못/없다 등 부정어가 후보에 있으면 반드시 부정 표현을 유지하라.\n"
)

_FEW_SHOT_TOPK_EXAMPLES = [
    (
        [
            [{"word": "배",     "score": 0.95}, {"word": "복통",  "score": 0.82}],
            [{"word": "아프다", "score": 0.90}, {"word": "통증",  "score": 0.85}],
            [{"word": "병원",   "score": 0.98}, {"word": "의원",  "score": 0.80}],
            [{"word": "가다",   "score": 0.92}, {"word": "방문",  "score": 0.75}],
        ],
        "배가 아파서 병원에 간다.",
    ),
    (
        [
            [{"word": "처방전", "score": 0.96}, {"word": "처방",  "score": 0.80}],
            [{"word": "받다",   "score": 0.93}, {"word": "수령",  "score": 0.70}],
            [{"word": "약국",   "score": 0.98}, {"word": "병원",  "score": 0.65}],
            [{"word": "가다",   "score": 0.94}, {"word": "방문",  "score": 0.78}],
        ],
        "처방전을 받고 약국에 간다.",
    ),
    (
        [
            [{"word": "알레르기", "score": 0.95}, {"word": "두드러기", "score": 0.70}],
            [{"word": "있다",     "score": 0.88}],
            [{"word": "약",       "score": 0.96}, {"word": "약물",    "score": 0.75}],
            [{"word": "못",       "score": 0.90}],
            [{"word": "먹다",     "score": 0.93}, {"word": "복용",    "score": 0.80}],
        ],
        "알레르기가 있어서 약을 못 먹는다.",
    ),
]


def _format_topk_positions(candidates: list[list[dict]]) -> str:
    """Format RAG candidates into the numbered-position text used in prompts."""
    lines = []
    for i, pos in enumerate(candidates, 1):
        if not pos:
            continue
        parts = " | ".join(f"{c['word']}({c['score']:.2f})" for c in pos)
        lines.append(f"위치 {i}: {parts}")
    return "\n".join(lines)


def build_topk_prompt(
    candidates: list[list[dict]],
    domain: str = "daily",
) -> str:
    """Build the few-shot prompt for top-K RAG candidates.

    Includes an explicit negation warning if any negation word appears in
    the candidate set.
    """
    examples = []
    for pos_candidates, sentence in _FEW_SHOT_TOPK_EXAMPLES:
        block = _format_topk_positions(pos_candidates)
        examples.append(f"{block}\n\n{sentence}")

    positions_text = _format_topk_positions(candidates)

    # Negation detection
    all_words      = {c["word"] for pos in candidates for c in pos}
    negation_found = all_words & _NEGATION_WORDS
    negation_warning = (
        f"\n⚠️ 부정어 '{', '.join(sorted(negation_found))}' 포함 — 모든 문장에 부정 표현 필수.\n"
        if negation_found else ""
    )

    return (
        TOPK_SYSTEM_INSTRUCTION
        + "\n### 예시 ###\n\n"
        + "\n\n---\n\n".join(examples)
        + "\n\n### 변환 ###\n"
        + f"도메인: {domain}\n"
        + negation_warning
        + positions_text
        + "\n\n"
    )


# ── Output parser ─────────────────────────────────────────────────────────────


def _parse_numbered_sentences(text: str) -> list[str]:
    """Extract '1. ...', '2. ...', '3. ...' sentences from model output."""
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^\d+[.)]\s+(.+)", line)
        if m:
            sentences.append(m.group(1).strip())
    return sentences


# ── Best-sentence selector ────────────────────────────────────────────────────


def _select_best_sentence(
    sentences: list[str],
    top1_words: list[str],
) -> str:
    """Pick the sentence that covers the most top-1 keywords.

    Falls back to sentences[0] if the list is empty or no word is matched.
    """
    if not sentences:
        return ""
    if not top1_words:
        return sentences[0]

    def _score(s: str) -> int:
        return sum(1 for w in top1_words if w in s)

    best = max(sentences, key=_score)
    return best


# ── SLLMService ───────────────────────────────────────────────────────────────


class SLLMService:
    """Lightweight sLLM service wrapping fine-tuned EXAONE-3.5 GGUF via llama-cpp-python.

    Parameters
    ----------
    model_path:
        Path to the GGUF file (downloaded via hf_hub_download at startup).
    n_threads:
        CPU thread count for llama.cpp inference (default: 4).
    """

    _STOP_TOKENS = ["[|endofturn|]", "[|user|]", "[|system|]"]

    _SYSTEM_MSG = (
        "너는 한국수어 번역 시스템의 후처리 sLLM이다. "
        "단어 시퀀스를 한국어 문장 하나로 변환하라.\n"
        "규칙:\n"
        "1. 반드시 해라체(-다, -ㄴ다, -는다)만 사용. -요/-습니다/-해/-야 금지.\n"
        "2. 과거 표현 없으면 현재형 사용. 가다→간다(O) 갔다(X)\n"
        "3. 입력에 없는 단어(매일,아침,함께,항상 등) 추가 금지.\n"
        "4. 설명 없이 문장 하나만 출력.\n"
        "5. 반드시 한국어만 출력. 영어 단어 사용 금지."
    )

    def __init__(self, model_path: str, n_threads: int = 4, n_gpu_layers: int = 0) -> None:
        from llama_cpp import Llama  # type: ignore

        # Patch Jinja2ChatFormatter before creating the Llama instance so that
        # EXAONE 4.0's embedded chat template (which uses {%- continue %}) works.
        self._patch_jinja2_loopcontrols()

        print(f"[sLLM] Loading GGUF: {model_path}  (n_gpu_layers={n_gpu_layers})")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,  # 0=CPU-only (Railway default); set >0 for Metal/CUDA
            verbose=False,
        )
        print("[sLLM] GGUF model ready")

    # ── Jinja2 patch (required for EXAONE 4.0 GGUF chat template) ────────────

    @staticmethod
    def _patch_jinja2_loopcontrols() -> None:
        """Patch llama_cpp Jinja2ChatFormatter to enable the loopcontrols extension.

        EXAONE 4.0 GGUF embeds a chat template that uses {%- continue %}, which
        jinja2 only supports with the loopcontrols extension. Since llama-cpp-python
        creates the jinja2 environment without this extension, we monkey-patch
        Jinja2ChatFormatter.__init__ to add it.
        """
        import jinja2
        import llama_cpp.llama_chat_format as _lcf  # type: ignore
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        if getattr(_lcf.Jinja2ChatFormatter, "_loopcontrols_patched", False):
            return

        def _patched(
            self,
            template: str,
            eos_token: str,
            bos_token: str,
            add_generation_prompt: bool = True,
            stop_token_ids: Optional[list[int]] = None,
        ) -> None:
            self.template               = template
            self.eos_token              = eos_token
            self.bos_token              = bos_token
            self.add_generation_prompt  = add_generation_prompt
            self.stop_token_ids        = set(stop_token_ids) if stop_token_ids else None
            self._environment = ImmutableSandboxedEnvironment(
                loader=jinja2.BaseLoader(),
                trim_blocks=True,
                lstrip_blocks=True,
                extensions=["jinja2.ext.loopcontrols"],
            ).from_string(self.template)

        _lcf.Jinja2ChatFormatter.__init__       = _patched
        _lcf.Jinja2ChatFormatter._loopcontrols_patched = True

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>...</think> reasoning blocks from EXAONE 4.0 output."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def _call_llm(self, prompt_text: str, max_tokens: int = 200) -> str:
        """Run greedy inference using GGUF-embedded chat template and return text."""
        result = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self._SYSTEM_MSG},
                {"role": "user",   "content": prompt_text},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            repeat_penalty=1.1,
            stop=self._STOP_TOKENS,
        )
        raw = result["choices"][0]["message"]["content"]
        return self._strip_think(raw)

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize_with_candidates(
        self,
        candidates: list[list[dict]],
    ) -> tuple[str, list[str]]:
        """Translate RAG top-K candidates into a Korean sentence.

        Parameters
        ----------
        candidates:
            Per-position candidate lists from FAISS search, e.g.
            ``[[{"word": "배", "score": 0.95}, ...], ...]``.

        Returns
        -------
        (best_sentence, [sentence_1, sentence_2, sentence_3])
            ``best_sentence`` is the sentence from the three generated ones
            that covers the most top-1 keywords.  If the model fails to
            produce 3 numbered sentences, the available sentences (or the
            raw output) are returned and ``best_sentence`` is sentence 1.
        """
        if not any(candidates):
            return "", []

        # Build prompt (includes few-shot examples + negation warning)
        prompt_text = build_topk_prompt(candidates)

        # Single greedy call → expected: one Korean sentence
        raw_output = self._call_llm(prompt_text, max_tokens=80)

        sentence = raw_output.strip().splitlines()[0].strip()
        return sentence, [sentence]
