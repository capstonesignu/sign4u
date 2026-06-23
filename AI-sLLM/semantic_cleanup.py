"""Layer 5: Semantic Cleanup.

Operations that modify semantics: tense rewrite, English replacement,
grammar fix, hallucination adverb removal.

Every operation produces a change_log entry for debugging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.cleanup_config import CleanupConfig
from medical_vocab import HALLUCINATION_ADVERBS


# ---------------------------------------------------------------------------
# Change log
# ---------------------------------------------------------------------------

@dataclass
class ChangeEntry:
    """A single semantic change applied to the output."""

    change_type: str   # "tense_rewrite", "english_replace", "grammar_fix", "hallucination_remove"
    before: str
    after: str

    def to_dict(self) -> dict:
        return {"type": self.change_type, "before": self.before, "after": self.after}


@dataclass
class SemanticCleanupResult:
    """Result of semantic cleanup with full change log."""

    text: str
    changes: list[ChangeEntry] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return len(self.changes) > 0


# ---------------------------------------------------------------------------
# English → Korean dictionary
# ---------------------------------------------------------------------------

ENGLISH_TO_KOREAN = {
    "needle": "바늘", "ice": "얼음", "pain": "통증", "fever": "발열",
    "headache": "두통", "cough": "기침", "cold": "감기", "blood": "혈액",
    "hospital": "병원", "doctor": "의사", "nurse": "간호사", "surgery": "수술",
    "medicine": "약", "pill": "알약", "tablet": "알약", "injection": "주사",
    "bandage": "붕대", "wound": "상처", "fracture": "골절", "cast": "깁스",
    "wheelchair": "휠체어", "crutch": "목발", "thermometer": "체온계",
    "stethoscope": "청진기", "syringe": "주사기", "gauze": "거즈",
    "ointment": "연고", "prescription": "처방전", "pharmacy": "약국",
    "head": "머리", "neck": "목", "shoulder": "어깨", "arm": "팔",
    "hand": "손", "finger": "손가락", "chest": "가슴", "stomach": "배",
    "back": "등", "waist": "허리", "leg": "다리", "knee": "무릎",
    "foot": "발", "toe": "발가락", "eye": "눈", "ear": "귀",
    "nose": "코", "mouth": "입", "tooth": "이빨", "tongue": "혀",
    "skin": "피부", "bone": "뼈", "muscle": "근육",
    "water": "물", "food": "음식", "sleep": "수면", "rest": "휴식",
    "test": "검사", "check": "확인", "appointment": "예약",
    "emergency": "응급", "ambulance": "구급차", "insurance": "보험",
}

# Allowed English abbreviations that should not be replaced
ENGLISH_WHITELIST = {"CT", "MRI", "X", "PET", "ICU", "ER", "IV", "OK"}


# ---------------------------------------------------------------------------
# Past → present tense rules
# ---------------------------------------------------------------------------

PAST_TENSE_MARKERS = {"어제", "지난", "이전", "작년", "그제", "그때", "예전"}

PAST_TO_PRESENT = [
    (r"했다\.", "합니다."), (r"했다$", "합니다."),
    (r"갔다\.", "갑니다."), (r"갔다$", "갑니다."),
    (r"왔다\.", "옵니다."), (r"왔다$", "옵니다."),
    (r"먹었다\.", "먹습니다."), (r"먹었다$", "먹습니다."),
    (r"마셨다\.", "마십니다."), (r"마셨다$", "마십니다."),
    (r"봤다\.", "봅니다."), (r"봤다$", "봅니다."),
    (r"보았다\.", "봅니다."), (r"보았다$", "봅니다."),
    (r"잤다\.", "잡니다."), (r"잤다$", "잡니다."),
    (r"쉬었다\.", "쉽니다."), (r"쉬었다$", "쉽니다."),
    (r"걸렸다\.", "걸립니다."), (r"걸렸다$", "걸립니다."),
    (r"참았다\.", "참습니다."), (r"참았다$", "참습니다."),
    (r"잡았다\.", "잡습니다."), (r"잡았다$", "잡습니다."),
    (r"받았다\.", "받습니다."), (r"받았다$", "받습니다."),
    (r"맞았다\.", "맞습니다."), (r"맞았다$", "맞습니다."),
    (r"찍었다\.", "찍습니다."), (r"찍었다$", "찍습니다."),
    (r"들었다\.", "듣습니다."), (r"들었다$", "듣습니다."),
    (r"적었다\.", "적습니다."), (r"적었다$", "적습니다."),
    (r"찜질했다\.", "찜질합니다."), (r"찜질했다$", "찜질합니다."),
    (r"부탁했다\.", "부탁합니다."), (r"부탁했다$", "부탁합니다."),
    (r"확인했다\.", "확인합니다."), (r"확인했다$", "확인합니다."),
    (r"제출했다\.", "제출합니다."), (r"제출했다$", "제출합니다."),
    (r"정리했다\.", "정리합니다."), (r"정리했다$", "정리합니다."),
    (r"요청했다\.", "요청합니다."), (r"요청했다$", "요청합니다."),
    (r"결정했다\.", "결정합니다."), (r"결정했다$", "결정합니다."),
    (r"조절했다\.", "조절합니다."), (r"조절했다$", "조절합니다."),
    (r"설정했다\.", "설정합니다."), (r"설정했다$", "설정합니다."),
    (r"깁스했다\.", "깁스합니다."), (r"깁스했다$", "깁스합니다."),
    (r"악수했다\.", "악수합니다."), (r"악수했다$", "악수합니다."),
]


# ---------------------------------------------------------------------------
# Grammar fixes (restricted scope: spacing, 조사, endings only)
# Does NOT touch: word order, subject insertion, semantic rewrite
# ---------------------------------------------------------------------------

GRAMMAR_FIXES = [
    # Wrong conjugations (ending normalization → 합쇼체)
    ("필요한다", "필요합니다"),
    ("필요하다", "필요합니다"),
    ("좋는다", "좋습니다"),
    ("많는다", "많습니다"),
    ("없는다", "없습니다"),
    ("있는다", "있습니다"),
    ("아프는다", "아픕니다"),
    ("기쁘는다", "기쁩니다"),
    ("슬프는다", "슬픕니다"),
    ("불편한다", "불편합니다"),
]


# ---------------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------------

class SemanticCleaner:
    """Semantic cleanup with change tracking."""

    def __init__(self, config: CleanupConfig | None = None):
        self.config = config or CleanupConfig()

    def clean(
        self, text: str, input_words: list[str] | None = None,
    ) -> SemanticCleanupResult:
        """Apply semantic cleanup operations with change logging.

        Each operation is guarded by a "do no harm" check: if the
        cleanup causes a keyword that was present before to disappear,
        the operation is rolled back and a 'rollback' entry is logged.
        """
        result = SemanticCleanupResult(text=text)

        if not text:
            return result

        words_set = set(input_words) if input_words else set()

        # Order matters: English first, then tense, grammar, hallucination
        if self.config.enable_english_replacement:
            result = self._safe_apply(result, self._replace_english, words_set)

        if self.config.enable_tense_rewrite and input_words:
            result = self._safe_apply(
                result, lambda r: self._rewrite_tense(r, input_words), words_set,
            )

        if self.config.enable_grammar_fix:
            result = self._safe_apply(result, self._fix_grammar, words_set)

        if self.config.enable_hallucination_removal and input_words:
            result = self._safe_apply(
                result, lambda r: self._remove_hallucinated_adverbs(r, input_words), words_set,
            )

        return result

    # --- Do-no-harm guard ---

    def _safe_apply(
        self,
        result: SemanticCleanupResult,
        operation,
        input_words: set[str],
    ) -> SemanticCleanupResult:
        """Apply an operation, but roll back if it loses keywords.

        Compares keyword coverage before/after the operation.
        If any keyword that was present is now missing, revert to
        the pre-operation text and log a rollback entry.
        """
        if not input_words:
            return operation(result)

        before_text = result.text
        before_changes = list(result.changes)  # snapshot
        before_coverage = self._keyword_coverage(before_text, input_words)

        new_result = operation(result)

        after_coverage = self._keyword_coverage(new_result.text, input_words)
        lost_keywords = before_coverage - after_coverage

        if lost_keywords:
            # Roll back: restore text and changes
            new_result.text = before_text
            new_result.changes = before_changes
            new_result.changes.append(ChangeEntry(
                "rollback",
                f"lost keywords: {', '.join(sorted(lost_keywords))}",
                "reverted to pre-cleanup text",
            ))

        return new_result

    @staticmethod
    def _keyword_coverage(text: str, input_words: set[str]) -> set[str]:
        """Return which input words are found in text (stem-aware)."""
        found = set()
        text_lower = text.lower().replace(" ", "")
        for w in input_words:
            wl = w.lower()
            if wl in text.lower():
                found.add(w)
            elif w.endswith("다") and len(w) > 1 and w[:-1] in text.lower():
                found.add(w)
            elif wl.replace(" ", "") in text_lower:
                found.add(w)
        return found

    # --- Private methods ---

    def _replace_english(self, result: SemanticCleanupResult) -> SemanticCleanupResult:
        text = result.text
        for eng, kor in ENGLISH_TO_KOREAN.items():
            pattern = r"(?<![a-zA-Z])" + re.escape(eng) + r"(?![a-zA-Z])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                new_text = re.sub(pattern, kor, text, flags=re.IGNORECASE)
                result.changes.append(ChangeEntry("english_replace", eng, kor))
                text = new_text
        result.text = re.sub(r"\s+", " ", text).strip()
        return result

    def _rewrite_tense(
        self, result: SemanticCleanupResult, input_words: list[str],
    ) -> SemanticCleanupResult:
        # Skip if input has past-tense markers
        if any(marker in input_words for marker in PAST_TENSE_MARKERS):
            return result

        text = result.text
        for pattern, replacement in PAST_TO_PRESENT:
            match = re.search(pattern, text)
            if match:
                original = match.group(0)
                new_text = re.sub(pattern, replacement, text)
                if new_text != text:
                    result.changes.append(ChangeEntry("tense_rewrite", original, replacement.rstrip(".")))
                    text = new_text
        result.text = text
        return result

    def _fix_grammar(self, result: SemanticCleanupResult) -> SemanticCleanupResult:
        text = result.text
        for wrong, correct in GRAMMAR_FIXES:
            if wrong in text:
                result.changes.append(ChangeEntry("grammar_fix", wrong, correct))
                text = text.replace(wrong, correct)
        result.text = text
        return result

    def _remove_hallucinated_adverbs(
        self, result: SemanticCleanupResult, input_words: list[str],
    ) -> SemanticCleanupResult:
        text = result.text
        for hw in HALLUCINATION_ADVERBS:
            if hw not in input_words:
                pattern = r"(?<![가-힣])" + re.escape(hw) + r"(?![가-힣])"
                if re.search(pattern, text):
                    result.changes.append(ChangeEntry("hallucination_remove", hw, ""))
                    text = re.sub(r"\s*" + pattern + r"\s*", " ", text).strip()
        result.text = re.sub(r"\s+", " ", text)
        return result
