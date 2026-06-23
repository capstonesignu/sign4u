"""Centralized medical/domain vocabulary for AI-sLLM.

Single source of truth for all medical entity sets used across:
- constraints.py  (runtime hallucination/keyword checks)
- evaluate.py     (offline evaluation metrics)
- postprocess.py  (hallucination word filtering)

When adding new terms, add them HERE — all other files import from this module.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Symptoms (증상)
# ---------------------------------------------------------------------------

SYMPTOM_KEYWORDS: set[str] = {
    "아프다", "통증", "열", "발열", "기침", "가래", "감기", "몸살",
    "어지럽다", "현기증", "두통", "복통", "설사", "변비", "구토",
    "토하다", "메스껍다", "소화", "쓰리다", "더부룩하다", "속쓰림",
    "오한", "떨리다", "춥다", "덥다", "식은땀", "호흡곤란", "숨",
    "쉬다", "힘들다", "가슴", "답답하다", "부종", "붓다", "발진",
    "가려움", "두드러기", "출혈", "피", "멍", "경련", "마비",
    "화상", "염좌", "골절", "염증", "감염", "부어오르다",
}

# ---------------------------------------------------------------------------
# Body parts (신체 부위)
# ---------------------------------------------------------------------------

BODY_PARTS: set[str] = {
    "머리", "얼굴", "목", "어깨", "팔", "손", "손가락", "가슴",
    "배", "복부", "허리", "엉덩이", "다리", "무릎", "발", "발가락",
    "눈", "코", "입", "귀", "혀", "이빨", "치아", "잇몸",
    "뇌", "심장", "폐", "간", "신장", "위", "장", "자궁",
    "피부", "뼈", "관절", "근육", "인대", "척추", "디스크",
    "오른쪽", "왼쪽", "상체", "하체", "전신",
}

# ---------------------------------------------------------------------------
# Tests / diagnostics (검사)
# ---------------------------------------------------------------------------

TEST_KEYWORDS: set[str] = {
    "혈액", "검사", "소변", "혈압", "혈당", "체온", "CT", "MRI",
    "X-ray", "엑스레이", "초음파", "내시경", "심전도", "뇌파",
    "채혈", "조영제", "생검", "조직검사", "PET", "촬영",
    "진단", "수치", "결과", "판독",
}

# ---------------------------------------------------------------------------
# Medication (약물)
# ---------------------------------------------------------------------------

MEDICATION_KEYWORDS: set[str] = {
    "약", "처방", "처방전", "약국", "진통제", "소염제", "항생제",
    "해열제", "감기약", "진해제", "거담제", "진정제", "수면제",
    "혈압약", "당뇨약", "소화제", "제산제", "연고", "가루약",
    "시럽", "주사", "수액", "알레르기", "부작용", "용량", "복용",
    "식전", "식후", "하루", "번", "알",
}

# ---------------------------------------------------------------------------
# Negation (부정)
# ---------------------------------------------------------------------------

NEGATION_WORDS: set[str] = {
    "없다", "없음", "안", "못", "못하다", "불가능", "불가",
    "필요없다", "필요 없다", "지 않다", "지 못하다",
}

# ---------------------------------------------------------------------------
# Composite sets (조합)
# ---------------------------------------------------------------------------

# Narrow: clearly clinical terms only — for hallucination detection.
# Excludes common body parts (손/발/입/눈) to avoid false positives.
MEDICAL_CLINICAL: set[str] = (
    SYMPTOM_KEYWORDS
    | TEST_KEYWORDS
    | MEDICATION_KEYWORDS
    | {"위", "장", "간", "심장", "폐", "신장"}  # internal organs only
)

# Broad: includes common body parts — for medical entity recall.
MEDICAL_BROAD: set[str] = MEDICAL_CLINICAL | {
    "머리", "얼굴", "목", "어깨", "팔", "손", "가슴", "배", "허리",
    "다리", "무릎", "발", "눈", "코", "입", "귀", "혀", "뼈", "관절",
    "피부", "근육", "피",
}

# Hallucination detection entity sets (used by constraints.py runtime checks).
# Same structure as MEDICAL_CLINICAL but organized by category for reporting.
HALLUCINATION_ENTITIES: dict[str, set[str]] = {
    "symptom": SYMPTOM_KEYWORDS,
    "test": TEST_KEYWORDS,
    "medication": MEDICATION_KEYWORDS,
    "internal_organ": {"뇌", "심장", "폐", "간", "신장", "위", "장", "자궁"},
}

# Words the model tends to hallucinate (adverbs/adjectives not in input).
# Used by postprocess.py to strip unauthorized additions.
HALLUCINATION_ADVERBS: list[str] = [
    "매일", "항상", "늘", "아침", "저녁", "함께", "아직",
    "정말", "매우", "너무", "많이", "새로운", "모든", "잘",
]
