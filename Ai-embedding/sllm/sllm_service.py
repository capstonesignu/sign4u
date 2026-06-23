"""Fine-tuned EXAONE-3.5-2.4B sLLM — 6-layer controlled inference pipeline.

Layer 1: Raw Generation      — GGUF Q4_K_M via llama-cpp-python (greedy)
Layer 2: Constraint Scoring  — keyword / negation / hallucination / length
Layer 3: Candidate Selection — safety-adjusted best + fallback
Layer 4: Format Cleanup      — prefix removal, sentence clipping
Layer 5: Semantic Cleanup    — grammar fix, English→Korean
Layer 6: Final Guardrail     — keyword/negation safety check + hard fallback

모델: AI-sLLM/exaone35-v5cons-merged/model_q4_k_m.gguf  (v5cons 합쇼체 학습)
훈련 데이터 포맷: "입력 단어: word1 / word2 / word3" (슬래시 구분)
출력: 합쇼체(-습니다/-ㅂ니다) 문장
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_GGUF_PATH = str(
    Path(__file__).parent.parent.parent
    / "AI-sLLM" / "exaone35-v5cons-merged" / "model_q4_k_m.gguf"
)

# ---------------------------------------------------------------------------
# EXAONE chat template
# ---------------------------------------------------------------------------

_TMPL_SYSTEM = "[|system|]{content}[|endofturn|]\n"
_TMPL_USER   = "[|user|]{content}\n"
_TMPL_ASST   = "[|assistant|]"

# v5cons 훈련 데이터와 동일한 system 메시지
_SYSTEM_MSG = (
    "단어 시퀀스를 자연스러운 한국어 문장 하나로 변환하라. "
    "반드시 합쇼체(-습니다, -ㅂ니다)만 사용하고, 입력에 없는 단어를 추가하지 마라. "
    "반드시 한국어만 출력하고 영어 단어를 사용하지 마라."
)


def _format_chat(user: str) -> str:
    return (
        _TMPL_SYSTEM.format(content=_SYSTEM_MSG)
        + _TMPL_USER.format(content=user)
        + _TMPL_ASST
    )


def _build_user_prompt(words: list[str]) -> str:
    return "입력 단어: " + " / ".join(words)


# ---------------------------------------------------------------------------
# Layer 2: Constraint scoring (vocabulary)
# ---------------------------------------------------------------------------

_NEGATION_WORDS: set[str] = {
    "없다", "없음", "안", "못", "못하다", "불가능", "불가",
    "필요없다", "필요 없다", "지 않다", "지 못하다",
}

_HALLUCINATION_ADVERBS: list[str] = [
    "매일", "항상", "늘", "아침", "저녁", "함께", "아직",
    "정말", "매우", "너무", "많이", "새로운", "모든", "잘",
]

_SYMPTOM_KW: set[str] = {
    "아프다", "통증", "열", "발열", "기침", "가래", "감기", "몸살",
    "어지럽다", "현기증", "두통", "복통", "설사", "변비", "구토",
    "토하다", "메스껍다", "소화", "쓰리다", "오한", "떨리다",
    "춥다", "덥다", "식은땀", "호흡곤란", "숨", "힘들다", "가슴",
    "답답하다", "부종", "붓다", "발진", "가려움", "두드러기",
    "출혈", "피", "멍", "경련", "마비", "화상", "염좌", "골절",
    "염증", "감염",
}
_TEST_KW: set[str] = {
    "혈액", "검사", "소변", "혈압", "혈당", "체온", "CT", "MRI",
    "엑스레이", "초음파", "내시경", "심전도", "채혈", "생검",
    "진단", "수치", "결과",
}
_MED_KW: set[str] = {
    "약", "처방", "처방전", "약국", "진통제", "소염제", "항생제",
    "해열제", "감기약", "진해제", "거담제", "진정제", "수면제",
    "혈압약", "당뇨약", "소화제", "연고", "가루약", "시럽", "주사",
    "수액", "알레르기", "부작용", "용량", "복용",
}
_ORGAN_KW: set[str] = {"뇌", "심장", "폐", "간", "신장", "위", "장", "자궁"}

_HALLUCINATION_ENTITIES: dict[str, set[str]] = {
    "symptom": _SYMPTOM_KW,
    "test": _TEST_KW,
    "medication": _MED_KW,
    "internal_organ": _ORGAN_KW,
}

_PARTICLES = set("이가을를은는도에서의와과로만부터까지한")
_FUNCTION_WORDS = {
    "나", "너", "우리", "그", "이", "저", "누구", "무엇",
    "언제", "어디", "어떻게", "왜", "안", "못", "네", "예",
}


def _is_independent(word: str, text: str) -> bool:
    for m in re.finditer(re.escape(word), text):
        s, e = m.start(), m.end()
        before = text[s - 1] if s > 0 else " "
        after  = text[e]     if e < len(text) else " "
        if "가" <= before <= "힣":
            continue
        if "가" <= after <= "힣" and after not in _PARTICLES:
            continue
        return True
    return False


@dataclass
class _Score:
    raw: str
    keyword: float = 0.0
    negation: float = 0.0
    hallucination: float = 0.0
    length: float = 0.0
    total: float = 0.0
    kw_ok: bool = False
    neg_ok: bool = False
    hal_ok: bool = False
    issues: list[str] = field(default_factory=list)


def _score(words: list[str], text: str) -> _Score:
    s = _Score(raw=text)

    # --- keyword ---
    content = {w for w in words if w not in _FUNCTION_WORDS}
    pred_l = text.lower()
    pred_ns = pred_l.replace(" ", "")
    missing = []
    for kw in content:
        kl = kw.lower()
        if kl in pred_l:
            continue
        if kw.endswith("다") and len(kw) > 1 and kw[:-1] in pred_l:
            continue
        if len(kl.replace(" ", "")) >= 2 and kl.replace(" ", "") in pred_ns:
            continue
        missing.append(kw)
    s.kw_ok   = len(missing) == 0
    s.keyword = 1.0 - (len(missing) / len(content)) if content else 1.0
    if missing:
        s.issues.append(f"missing_kw:{missing}")

    # --- negation ---
    inp_joined = " ".join(words)
    has_neg = any(nw in inp_joined for nw in _NEGATION_WORDS)
    if not has_neg:
        s.negation = 1.0
        s.neg_ok   = True
    else:
        neg_word = next((nw for nw in _NEGATION_WORDS if nw in inp_joined), None)
        bad = False
        if neg_word:
            if ("없다" in inp_joined or "없음" in inp_joined) and "있다" in pred_l:
                bad = True
            if ("못" in inp_joined or "안" in inp_joined) and ("가능" in pred_l or "할 수" in pred_l):
                bad = True
            root = neg_word.replace(" ", "").replace("없다", "없").replace("못하다", "못")
            if not bad and root not in pred_l.replace(" ", ""):
                bad = True
        s.neg_ok   = not bad
        s.negation = 0.0 if bad else 1.0
        if bad:
            s.issues.append("negation_flip")

    # --- hallucination ---
    inp_lower = inp_joined.lower()
    inp_stems = set(words)
    for w in words:
        if w.endswith("다") and len(w) > 1:
            inp_stems.add(w[:-1])
    hals = []
    for cat, ents in _HALLUCINATION_ENTITIES.items():
        for e in ents:
            el = e.strip().lower()
            if not el or el in inp_lower or el in inp_stems:
                continue
            if _is_independent(el, pred_l):
                hals.append(e)
    adv_hals = [h for h in _HALLUCINATION_ADVERBS
                if h not in words and re.search(r"(?<![가-힣])" + re.escape(h) + r"(?![가-힣])", text)]
    total_issues = len(hals) + len(adv_hals)
    s.hal_ok       = total_issues == 0
    s.hallucination = max(0.0, 1.0 - total_issues * 0.3)
    if hals or adv_hals:
        s.issues.append(f"hallucination:{hals + adv_hals}")

    # --- length ---
    n = len(text)
    if n == 0:
        s.length = 0.0
        s.issues.append("empty")
    elif n > 120:
        s.length = max(0.0, 1.0 - (n - 120) / 100)
        s.issues.append("too_long")
    elif n < 3:
        s.length = 0.3
    else:
        s.length = 1.0

    # weighted total
    s.total = (0.35 * s.keyword + 0.25 * s.negation
               + 0.25 * s.hallucination + 0.15 * s.length)
    return s


def _select_best(scored: list[_Score]) -> tuple[_Score, _Score | None]:
    def adj(s: _Score) -> float:
        v = s.total
        if s.kw_ok:  v += 0.2
        if s.neg_ok: v += 0.2
        if s.hal_ok: v += 0.05
        return v
    ranked = sorted(scored, key=adj, reverse=True)
    return ranked[0], (ranked[1] if len(ranked) > 1 else None)


# ---------------------------------------------------------------------------
# Layer 4: Format cleanup
# ---------------------------------------------------------------------------

_PREFIX_RE = [
    re.compile(r"^출력\s*문장\s*[:：]\s*", re.I),
    re.compile(r"^출력\s*[:：]\s*",         re.I),
    re.compile(r"^정답\s*[:：]\s*",          re.I),
    re.compile(r"^문장\s*[:：]\s*",          re.I),
    re.compile(r"^답변\s*[:：]\s*",          re.I),
]


def _format_clean(text: str) -> str:
    if not text:
        return ""
    text = text.strip().replace("```", "").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    text = lines[0]
    for pat in _PREFIX_RE:
        text = pat.sub("", text).strip()
    text = text.strip("\"'""''")
    m = re.search(r"(.+?[.!?。])", text)
    if m:
        text = m.group(1).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if text and text[-1] not in ".!?。":
        text += "."
    return text.strip()


# ---------------------------------------------------------------------------
# Layer 5: Semantic cleanup
# ---------------------------------------------------------------------------

_ENGLISH_TO_KO = {
    "needle": "바늘", "pain": "통증", "fever": "발열",
    "headache": "두통", "cough": "기침", "cold": "감기", "blood": "혈액",
    "hospital": "병원", "doctor": "의사", "nurse": "간호사", "surgery": "수술",
    "medicine": "약", "pill": "알약", "injection": "주사",
    "bandage": "붕대", "wound": "상처", "fracture": "골절",
    "ointment": "연고", "prescription": "처방전", "pharmacy": "약국",
    "head": "머리", "neck": "목", "shoulder": "어깨", "arm": "팔",
    "chest": "가슴", "stomach": "배", "back": "등", "waist": "허리",
    "leg": "다리", "knee": "무릎", "foot": "발", "eye": "눈",
    "ear": "귀", "nose": "코", "mouth": "입", "tooth": "이빨",
    "skin": "피부", "bone": "뼈", "muscle": "근육",
    "test": "검사", "appointment": "예약", "emergency": "응급",
}
_ENGLISH_WHITELIST = {"CT", "MRI", "X", "PET", "ICU", "ER", "IV", "OK"}

_GRAMMAR_FIXES = [
    # 해라체 잔재 → 합쇼체
    ("필요합니다다", "필요합니다"),
    ("필요한다", "필요합니다"),
    ("좋는다", "좋습니다"),
    ("많는다", "많습니다"),
    ("없는다", "없습니다"),
    ("있는다", "있습니다"),
    ("아프는다", "아픕니다"),
    ("불편한다", "불편합니다"),
    # 이중 어미 제거
    ("습니다습니다", "습니다"),
    ("ㅂ니다ㅂ니다", "ㅂ니다"),
]


def _semantic_clean(text: str, words: list[str]) -> str:
    if not text:
        return text

    # English replacement
    for eng, kor in _ENGLISH_TO_KO.items():
        if eng.upper() in _ENGLISH_WHITELIST:
            continue
        pat = r"(?<![a-zA-Z])" + re.escape(eng) + r"(?![a-zA-Z])"
        text = re.sub(pat, kor, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()

    # Grammar fixes
    for wrong, correct in _GRAMMAR_FIXES:
        text = text.replace(wrong, correct)

    # Remove hallucinated adverbs
    for hw in _HALLUCINATION_ADVERBS:
        if hw not in words:
            pat = r"\s*(?<![가-힣])" + re.escape(hw) + r"(?![가-힣])\s*"
            text = re.sub(pat, " ", text).strip()

    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Layer 6: Final guardrail
# ---------------------------------------------------------------------------

def _guardrail(text: str, words: list[str]) -> tuple[bool, list[str]]:
    """Returns (passed, warnings)."""
    warnings = []
    if not text or not text.strip():
        return False, ["empty_output"]
    if len(text) > 120:
        warnings.append("too_long")
    if any(m in text for m in ["입력 단어", "출력 문장", "예시", "규칙", "단어:"]):
        return False, ["prompt_residue"]
    # 비정상 패턴: 다다., 있으면 있다, 이중 합쇼체
    if (re.search(r"다다[.!?]?$", text)
            or "있으면 있다" in text
            or re.search(r"습니다습니다", text)):
        return False, ["malformed_output"]

    content = {w for w in words if w not in _FUNCTION_WORDS}
    tl = text.lower()
    for kw in content:
        kl = kw.lower()
        found = (kl in tl
                 or (kw.endswith("다") and len(kw) > 1 and kw[:-1] in tl)
                 or kl.replace(" ", "") in tl.replace(" ", ""))
        if not found:
            return False, [f"final_kw_missing:{kw}"]

    inp_joined = " ".join(words)
    if any(nw in inp_joined for nw in _NEGATION_WORDS):
        neg_found = any(
            nw.replace("없다", "없").replace("못하다", "못").replace(" ", "") in tl.replace(" ", "")
            for nw in _NEGATION_WORDS if nw in inp_joined
        )
        if not neg_found:
            return False, ["negation_missing"]

    return len(warnings) == 0, warnings


# ---------------------------------------------------------------------------
# Hard fallback: 합쇼체 문장 생성
# ---------------------------------------------------------------------------

# 불규칙 활용 동사 (받침 없는 어간 + ㅂ니다)
_VERB_POLITE_MAP = {
    "가다": "갑니다",   "오다": "옵니다",
    "하다": "합니다",   "되다": "됩니다",
    "보다": "봅니다",   "주다": "줍니다",
    "두다": "둡니다",   "서다": "섭니다",
    "쓰다": "씁니다",   "크다": "큽니다",
    "아프다": "아픕니다",  "나쁘다": "나쁩니다",
    "바쁘다": "바쁩니다",  "슬프다": "슬픕니다",
}


def _make_fallback_sentence(words: list[str]) -> str:
    """단어 리스트 → 합쇼체 문장 (모델 실패 시 규칙 기반 fallback)."""
    if not words:
        return ""
    last = words[-1]
    rest = words[:-1]
    if last.endswith("다"):
        polite = _VERB_POLITE_MAP.get(last, last[:-1] + "습니다")
    else:
        polite = last + "입니다"
    parts = rest + [polite]
    return " ".join(parts) + "."


# ---------------------------------------------------------------------------
# FinetunedSLLMService
# ---------------------------------------------------------------------------

class FinetunedSLLMService:
    """EXAONE-3.5-2.4B Q4_K_M — 6-layer pipeline (llama-cpp-python CPU)."""

    def __init__(
        self,
        gguf_path: str | None = None,
        n_threads: int = 4,
        n_ctx: int = 1024,
        adapter_path: str | None = None,   # 하위 호환, 무시
        base_model_path: str | None = None, # 하위 호환, 무시
    ) -> None:
        from llama_cpp import Llama
        model_path = gguf_path or DEFAULT_GGUF_PATH
        print(f"[sLLM] loading GGUF: {model_path}")
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
        )
        print(f"[sLLM] ready  (llama-cpp CPU, Q4_K_M, threads={n_threads})")

    # --- Layer 1: raw generation ---

    def _generate(self, user_prompt: str, temperature: float = 0.0) -> str:
        prompt = _format_chat(user_prompt)
        print(f"USER: {user_prompt}.\n")
        out = self._llm(
            prompt,
            max_tokens=80,
            echo=False,
            stop=["[|endofturn|]", "[|user|]", "[|system|]"],
            repeat_penalty=1.1,
            temperature=temperature,
        )
        return out["choices"][0]["text"].strip()

    # --- Full pipeline ---

    def normalize_with_candidates(
        self,
        candidates: list[list[dict]],
    ) -> tuple[str, list[str]]:
        if not any(candidates):
            return "", []

        top1_words = [pos[0]["word"] for pos in candidates if pos]

        # 연속 중복 제거 (춥다/춥다 → 춥다)
        deduped: list[str] = []
        for w in top1_words:
            if not deduped or deduped[-1] != w:
                deduped.append(w)
        user_prompt = _build_user_prompt(deduped)

        # Layer 1: greedy generation (deterministic)
        raw_greedy = self._generate(user_prompt, temperature=0.0)

        if not raw_greedy:
            result = _make_fallback_sentence(deduped)
            return result, [result]

        # Layer 2: score
        scored = [_score(top1_words, raw_greedy)]

        # Layer 3: select best (single candidate)
        best, _ = _select_best(scored)

        # Layer 4+5: cleanup pipeline
        def _clean(text: str) -> str:
            t = _format_clean(text)
            t = _semantic_clean(t, top1_words)
            return t

        best_clean = _clean(best.raw)

        # Layer 6: guardrail
        passed, _ = _guardrail(best_clean, top1_words)
        result = best_clean if passed else None

        # Hard fallback: guardrail 실패 시 규칙 기반 합쇼체 문장
        if not result or any(m in result for m in ["입력 단어", "출력 문장", "예시"]):
            result = _make_fallback_sentence(deduped)

        return result, [result]
