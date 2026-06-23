"""Post-processing utilities for sLLM output."""

from __future__ import annotations

import re


PREFIX_PATTERNS = [
    r"^출력\s*문장\s*[:：]\s*",
    r"^출력\s*[:：]\s*",
    r"^정답\s*[:：]\s*",
    r"^문장\s*[:：]\s*",
    r"^답변\s*[:：]\s*",
]

# ---------------------------------------------------------------------------
# Polite → plain form conversion (존댓말 → 해라체)
# ---------------------------------------------------------------------------

# Ending replacements: longest match first to avoid partial matches
POLITE_TO_PLAIN = [
    # -습니다 forms
    (r"습니다\.", "다."),
    (r"습니다\?", "니?"),
    (r"습니다$", "다."),
    (r"합니다\.", "한다."),
    (r"합니다\?", "하니?"),
    (r"합니다$", "한다."),
    # -ㅂ니다
    (r"됩니다\.", "된다."),
    (r"됩니다$", "된다."),
    (r"갑니다\.", "간다."),
    (r"갑니다$", "간다."),
    (r"옵니다\.", "온다."),
    (r"옵니다$", "온다."),
    # -아/어요 forms (common)
    (r"해요\.", "한다."),
    (r"해요\?", "하니?"),
    (r"해요$", "한다."),
    (r"돼요\.", "된다."),
    (r"돼요$", "된다."),
    (r"가요\.", "간다."),
    (r"가요\?", "가니?"),
    (r"가요$", "간다."),
    (r"와요\.", "온다."),
    (r"와요$", "온다."),
    (r"봐요\.", "본다."),
    (r"봐요$", "본다."),
    (r"줘요\.", "준다."),
    (r"줘요$", "준다."),
    # -어/아요 ending pattern
    (r"어요\.", "는다."),
    (r"어요\?", "니?"),
    (r"어요$", "는다."),
    (r"아요\.", "는다."),
    (r"아요\?", "니?"),
    (r"아요$", "는다."),
    # Specific common words
    (r"필요해\.", "필요하다."),
    (r"필요해$", "필요하다."),
    (r"좋아해\.", "좋아한다."),
    (r"좋아해$", "좋아한다."),
    (r"싫어해\.", "싫어한다."),
    (r"싫어해$", "싫어한다."),
    # -해 (informal)
    (r"인해\.", "인하다."),
    # Generic -ㅂ니까/-나요 question forms
    (r"나요\?", "니?"),
    (r"나요\.", "니?"),
    (r"까요\?", "니?"),
    (r"까요\.", "니?"),
    # -세요/-시 honorific
    (r"가세요\.", "간다."),
    (r"가세요$", "간다."),
    (r"오세요\.", "온다."),
    (r"오세요$", "온다."),
    (r"하세요\.", "한다."),
    (r"하세요$", "한다."),
    (r"드세요\.", "먹는다."),
    (r"드세요$", "먹는다."),
    (r"가신다\.", "간다."),
    (r"가신다$", "간다."),
    (r"오신다\.", "온다."),
    (r"오신다$", "온다."),
    (r"하신다\.", "한다."),
    (r"하신다$", "한다."),
    # -겠- future/intention → present
    (r"가겠다\.", "간다."),
    (r"가겠다$", "간다."),
    (r"자겠다\.", "잔다."),
    (r"자겠다$", "잔다."),
    (r"하겠다\.", "한다."),
    (r"하겠다$", "한다."),
    (r"먹겠다\.", "먹는다."),
    (r"먹겠다$", "먹는다."),
    (r"마시겠다\.", "마신다."),
    (r"마시겠다$", "마신다."),
]

# ---------------------------------------------------------------------------
# Past → present tense conversion (과거형 → 현재형)
# Only applied when input has NO past-tense markers (어제, 지난, 했, 였, 었)
# ---------------------------------------------------------------------------

PAST_TENSE_MARKERS = {"어제", "지난", "이전", "작년", "그제", "그때", "예전"}

PAST_TO_PRESENT = [
    # -했다 → -한다
    (r"했다\.", "한다."),
    (r"했다$", "한다."),
    # -갔다 → -간다
    (r"갔다\.", "간다."),
    (r"갔다$", "간다."),
    # -왔다 → -온다
    (r"왔다\.", "온다."),
    (r"왔다$", "온다."),
    # -먹었다 → -먹는다
    (r"먹었다\.", "먹는다."),
    (r"먹었다$", "먹는다."),
    # -마셨다 → -마신다
    (r"마셨다\.", "마신다."),
    (r"마셨다$", "마신다."),
    # -봤다/보았다 → -본다
    (r"봤다\.", "본다."),
    (r"봤다$", "본다."),
    (r"보았다\.", "본다."),
    (r"보았다$", "본다."),
    # -잤다 → -잔다
    (r"잤다\.", "잔다."),
    (r"잤다$", "잔다."),
    # -쉬었다 → -쉰다
    (r"쉬었다\.", "쉰다."),
    (r"쉬었다$", "쉰다."),
    # -걸렸다 → -걸린다
    (r"걸렸다\.", "걸린다."),
    (r"걸렸다$", "걸린다."),
    # -참았다 → -참는다
    (r"참았다\.", "참는다."),
    (r"참았다$", "참는다."),
    # -잡았다 → -잡는다
    (r"잡았다\.", "잡는다."),
    (r"잡았다$", "잡는다."),
    # Generic -았다/-었다 endings (catch-all, lower priority)
    # -았다 → -(ㄴ/는)다  (applied to full sentence ending only)
    (r"받았다\.", "받는다."),
    (r"받았다$", "받는다."),
    (r"맞았다\.", "맞는다."),
    (r"맞았다$", "맞는다."),
    (r"찍었다\.", "찍는다."),
    (r"찍었다$", "찍는다."),
    (r"들었다\.", "듣는다."),
    (r"들었다$", "듣는다."),
    (r"적었다\.", "적는다."),
    (r"적었다$", "적는다."),
    # -했다 in compound verbs (middle of sentence)
    (r"찜질했다\.", "찜질한다."),
    (r"찜질했다$", "찜질한다."),
    (r"악수했다\.", "악수한다."),
    (r"악수했다$", "악수한다."),
    (r"부탁했다\.", "부탁한다."),
    (r"부탁했다$", "부탁한다."),
    (r"깁스했다\.", "깁스한다."),
    (r"깁스했다$", "깁스한다."),
    (r"설정했다\.", "설정한다."),
    (r"설정했다$", "설정한다."),
    (r"요청했다\.", "요청한다."),
    (r"요청했다$", "요청한다."),
    (r"결정했다\.", "결정한다."),
    (r"결정했다$", "결정한다."),
    (r"확인했다\.", "확인한다."),
    (r"확인했다$", "확인한다."),
    (r"조절했다\.", "조절한다."),
    (r"조절했다$", "조절한다."),
    (r"제출했다\.", "제출한다."),
    (r"제출했다$", "제출한다."),
    (r"정리했다\.", "정리한다."),
    (r"정리했다$", "정리한다."),
]


def convert_past_to_present(text: str, input_words: list[str] | None = None) -> str:
    """Convert past tense to present tense when input has no past-tense markers."""
    if input_words:
        # If input contains past-tense markers, keep past tense
        if any(marker in input_words for marker in PAST_TENSE_MARKERS):
            return text
    result = text
    for pattern, replacement in PAST_TO_PRESENT:
        result = re.sub(pattern, replacement, result)
    return result


# Words that should NOT appear if they weren't in input
from medical_vocab import HALLUCINATION_ADVERBS as HALLUCINATION_WORDS

# English→Korean replacement for leaked English words in model output
ENGLISH_TO_KOREAN = {
    # Medical / body
    "needle": "바늘", "ice": "얼음", "pain": "통증", "fever": "발열",
    "headache": "두통", "cough": "기침", "cold": "감기", "blood": "혈액",
    "hospital": "병원", "doctor": "의사", "nurse": "간호사", "surgery": "수술",
    "medicine": "약", "pill": "알약", "tablet": "알약", "injection": "주사",
    "bandage": "붕대", "wound": "상처", "fracture": "골절", "cast": "깁스",
    "wheelchair": "휠체어", "crutch": "목발", "thermometer": "체온계",
    "stethoscope": "청진기", "syringe": "주사기", "gauze": "거즈",
    "ointment": "연고", "prescription": "처방전", "pharmacy": "약국",
    # Body parts
    "head": "머리", "neck": "목", "shoulder": "어깨", "arm": "팔",
    "hand": "손", "finger": "손가락", "chest": "가슴", "stomach": "배",
    "back": "등", "waist": "허리", "leg": "다리", "knee": "무릎",
    "foot": "발", "toe": "발가락", "eye": "눈", "ear": "귀",
    "nose": "코", "mouth": "입", "tooth": "이빨", "tongue": "혀",
    "skin": "피부", "bone": "뼈", "muscle": "근육",
    # Actions / general
    "water": "물", "food": "음식", "sleep": "수면", "rest": "휴식",
    "test": "검사", "check": "확인", "appointment": "예약",
    "emergency": "응급", "ambulance": "구급차", "insurance": "보험",
}

# Common grammatical errors the model produces
GRAMMAR_FIXES = [
    # Wrong conjugations
    ("필요한다", "필요하다"),
    ("좋는다", "좋다"),
    ("많는다", "많다"),
    ("없는다", "없다"),
    ("있는다", "있다"),
    ("아프는다", "아프다"),
    ("기쁘는다", "기쁘다"),
    ("슬프는다", "슬프다"),
    ("재밌었는다", "재미있었다"),
    ("걸렸는다", "걸렸다"),
    # Truncated words (tokenizer artifacts)
    ("나는 오 ", "나는 오늘 "),
    ("오 날씨", "오늘 날씨"),
    ("오 시험", "오늘 시험"),
]


def convert_polite_to_plain(text: str) -> str:
    """Convert polite/honorific endings to plain/declarative form."""
    result = text
    for pattern, replacement in POLITE_TO_PLAIN:
        result = re.sub(pattern, replacement, result)
    return result


def fix_grammar(text: str, input_words: list[str] | None = None) -> str:
    """Fix common grammatical errors produced by the model."""
    result = text

    # Fix truncated words: model sometimes outputs partial tokens
    if input_words:
        # "오" → "오늘" when 오늘 is in input but not in output
        if "오늘" in input_words and "오늘" not in result and "오 " in result:
            result = result.replace("오 ", "오늘 ", 1)

    for wrong, correct in GRAMMAR_FIXES:
        if wrong in result:
            result = result.replace(wrong, correct)

    # Fix "말라요" → "마르다" style
    if input_words:
        result = re.sub(r'말라요\.?$', '마르다.', result)

    return result


def replace_english_leaks(text: str) -> str:
    """Replace leaked English words with Korean equivalents.

    Two-layer approach:
    1. Known dictionary: exact replacement for common medical/body terms
    2. Auto-detect: find ANY remaining English words and replace from
       input context or remove them entirely

    The base model (EXAONE) is multilingual and occasionally outputs
    English words instead of Korean, especially for low-frequency terms.
    """
    result = text

    # Layer 1: Known dictionary replacement
    for eng, kor in ENGLISH_TO_KOREAN.items():
        pattern = r'(?<![a-zA-Z])' + re.escape(eng) + r'(?![a-zA-Z])'
        result = re.sub(pattern, kor, result, flags=re.IGNORECASE)

    # Layer 2: Auto-detect remaining English — only log, don't delete
    # Deleting unknown English risks losing critical medical information
    # (e.g., "nausea가 심하다" — removing "nausea" loses the symptom)
    english_words = re.findall(r'[a-zA-Z]{2,}', result)
    for ew in english_words:
        if ew.upper() in {"CT", "MRI", "X", "PET", "ICU", "ER", "IV", "OK"}:
            continue
        # Log warning but keep the word — safer than deleting
        import logging
        logging.warning(f"[postprocess] unknown English word '{ew}' in output: {result}")

    result = re.sub(r'\s+', ' ', result).strip()
    return result


def remove_hallucinated_words(text: str, input_words: list[str] | None = None) -> str:
    """Remove common hallucinated adverbs/words not in the input."""
    if input_words is None:
        return text
    for hw in HALLUCINATION_WORDS:
        if hw not in input_words:
            # Use word boundary to avoid matching inside other words
            # e.g. "늘" should not match inside "오늘"
            pattern = r"(?<![가-힣])" + re.escape(hw) + r"(?![가-힣])"
            if re.search(pattern, text):
                text = re.sub(r"\s*" + pattern + r"\s*", " ", text).strip()
    # Clean up double spaces
    text = re.sub(r"\s+", " ", text)
    return text


def clean_model_output(text: str, input_words: list[str] | None = None) -> str:
    """Extract a single clean Korean sentence from raw model output."""
    if not text:
        return ""

    cleaned = str(text).strip()
    cleaned = cleaned.replace("```", "").strip()

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]

    for pattern in PREFIX_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

    cleaned = cleaned.strip("\"'""''")
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Keep only the first sentence when the model gives extra explanation.
    match = re.search(r"(.+?[.!?。])", cleaned)
    if match:
        cleaned = match.group(1).strip()

    if cleaned and cleaned[-1] not in ".!?。":
        cleaned += "."

    # Post-processing pipeline
    cleaned = replace_english_leaks(cleaned)          # English→Korean first
    cleaned = convert_polite_to_plain(cleaned)
    cleaned = convert_past_to_present(cleaned, input_words)
    cleaned = fix_grammar(cleaned, input_words)
    cleaned = remove_hallucinated_words(cleaned, input_words)

    return cleaned


def validate_sentence(text: str) -> tuple[bool, str]:
    """Return whether the final normalized sentence is acceptable."""
    if not text:
        return False, "empty output"
    if len(text) > 120:
        return False, "output too long"
    if any(marker in text for marker in ["입력", "출력", "예시", "규칙"]):
        return False, "output contains prompt residue"
    return True, ""
