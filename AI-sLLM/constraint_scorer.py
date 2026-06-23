"""Layer 2: Constraint Scoring.

Scores each raw candidate on multiple dimensions.
Does NOT select — only evaluates. Selection is in candidate_selector.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.scoring_config import ScoringConfig
from failure_taxonomy import FailureType
from medical_vocab import (
    NEGATION_WORDS,
    HALLUCINATION_ENTITIES,
    HALLUCINATION_ADVERBS,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateScore:
    """Score report for a single candidate."""

    raw_text: str
    keyword_score: float = 0.0
    negation_score: float = 0.0
    hallucination_score: float = 0.0
    length_score: float = 0.0
    total_score: float = 0.0

    # Safety flags (used by candidate_selector for fallback ranking)
    all_keywords_preserved: bool = False
    negation_consistent: bool = False
    no_medical_hallucination: bool = False

    # Detected failures
    failure_types: list[FailureType] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    # Reserved interfaces (not implemented yet)
    # semantic_consistency_score: float = 0.0
    # contradiction_score: float = 0.0


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

# Korean particles that follow nouns — same as original constraints.py
_PARTICLES = set("이가을를은는도에서의와과로만부터까지한")


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,.!?。]+", text) if t]


def _extract_content_words(words: list[str]) -> set[str]:
    """Return content words from input (excluding function words)."""
    function_words = {
        "나", "너", "우리", "그", "이", "저", "누구", "무엇",
        "언제", "어디", "어떻게", "왜", "안", "못", "네", "예",
    }
    return {w for w in words if w not in function_words}


def _is_independent_word(word: str, text: str) -> bool:
    """Check if word appears as an independent unit (not substring)."""
    for match in re.finditer(re.escape(word), text):
        start, end = match.start(), match.end()
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "
        before_is_korean = "가" <= before <= "힣"
        after_is_korean = "가" <= after <= "힣"
        if before_is_korean:
            continue
        if after_is_korean and after not in _PARTICLES:
            continue
        return True
    return False


# ---------------------------------------------------------------------------
# Individual scoring functions
# ---------------------------------------------------------------------------

def score_keyword_preservation(
    input_words: list[str], prediction: str,
) -> tuple[float, bool, list[str]]:
    """Score keyword preservation (0.0 to 1.0).

    Returns (score, all_preserved, missing_keywords).
    """
    keywords = _extract_content_words(input_words)
    if not keywords:
        return 1.0, True, []

    pred_lower = prediction.lower()
    pred_nospace = pred_lower.replace(" ", "")

    missing: list[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        # Direct match
        if kw_lower in pred_lower:
            continue
        # Stem match (가다 → 가)
        if kw.endswith("다") and len(kw) > 1:
            stem = kw[:-1]
            if stem in pred_lower:
                continue
        # Spacing variation
        kw_nospace = kw_lower.replace(" ", "")
        if len(kw_nospace) >= 2 and kw_nospace in pred_nospace:
            continue
        missing.append(kw)

    if not keywords:
        return 1.0, True, []
    score = 1.0 - len(missing) / len(keywords)
    return score, len(missing) == 0, missing


def score_negation_preservation(
    input_words: list[str], prediction: str,
) -> tuple[float, bool, str]:
    """Score negation preservation (0.0 or 1.0).

    Returns (score, consistent, reason).
    """
    input_joined = " ".join(input_words)
    has_negation = any(nw in input_joined for nw in NEGATION_WORDS)
    if not has_negation:
        return 1.0, True, ""

    pred_lower = prediction.lower()

    # Find which negation word
    neg_word = None
    for nw in NEGATION_WORDS:
        if nw in input_joined:
            neg_word = nw
            break

    # Check for reversal: input has negation but output has affirmation
    if neg_word:
        if ("없다" in input_joined or "없음" in input_joined) and "있다" in pred_lower:
            return 0.0, False, f"negation_reversed: '{neg_word}' → affirmative"
        if ("못" in input_joined or "안" in input_joined) and ("가능" in pred_lower or "할 수" in pred_lower):
            return 0.0, False, f"negation_reversed: '{neg_word}' → positive"

    # Check negation word presence
    if neg_word and neg_word not in pred_lower:
        root = neg_word.replace(" ", "").replace("없다", "없").replace("못하다", "못")
        if root not in pred_lower.replace(" ", ""):
            return 0.0, False, f"negation_missing: '{neg_word}' not in output"

    return 1.0, True, ""


def score_hallucination(
    input_words: list[str], prediction: str,
) -> tuple[float, bool, list[dict]]:
    """Score hallucination (0.0 to 1.0).

    Checks for medical entities in output that are not in input.
    Returns (score, clean, hallucinated_items).
    """
    input_text = " ".join(input_words).lower()
    input_stems = set(input_words)
    for w in input_words:
        if w.endswith("다") and len(w) > 1:
            input_stems.add(w[:-1])

    hallucinations: list[dict] = []
    pred_text = prediction.lower()

    for category, entities in HALLUCINATION_ENTITIES.items():
        for entity in entities:
            e = entity.strip().lower()
            if not e or e in input_text or e in input_stems:
                continue
            if _is_independent_word(e, pred_text):
                hallucinations.append({"category": category, "entity": entity.strip()})

    # Also check hallucinated adverbs
    adverb_hallucinations = []
    for hw in HALLUCINATION_ADVERBS:
        if hw not in input_words:
            pattern = r"(?<![가-힣])" + re.escape(hw) + r"(?![가-힣])"
            if re.search(pattern, prediction):
                adverb_hallucinations.append(hw)

    total_issues = len(hallucinations) + len(adverb_hallucinations)
    score = max(0.0, 1.0 - total_issues * 0.3)  # each issue costs 0.3

    all_items = hallucinations + [{"category": "adverb", "entity": a} for a in adverb_hallucinations]
    return score, total_issues == 0, all_items


def score_length(prediction: str) -> tuple[float, list[FailureType]]:
    """Score output length (0.0 to 1.0)."""
    failures = []
    if not prediction.strip():
        return 0.0, [FailureType.EMPTY_OUTPUT]
    length = len(prediction)
    if length > 120:
        failures.append(FailureType.OVER_GENERATION)
        return max(0.0, 1.0 - (length - 120) / 100), failures
    if length < 3:
        return 0.3, failures
    return 1.0, failures


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class ConstraintScorer:
    """Scores raw candidates on constraint dimensions."""

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()

    def score_candidate(
        self, input_words: list[str], raw_text: str,
    ) -> CandidateScore:
        """Score a single raw candidate."""
        cs = CandidateScore(raw_text=raw_text)

        # Keyword
        kw_score, kw_all, kw_missing = score_keyword_preservation(input_words, raw_text)
        cs.keyword_score = kw_score
        cs.all_keywords_preserved = kw_all
        if not kw_all:
            cs.failure_types.append(FailureType.KEYWORD_OMISSION)
            cs.details["missing_keywords"] = kw_missing

        # Negation
        neg_score, neg_ok, neg_reason = score_negation_preservation(input_words, raw_text)
        cs.negation_score = neg_score
        cs.negation_consistent = neg_ok
        if not neg_ok:
            cs.failure_types.append(FailureType.NEGATION_FLIP)
            cs.details["negation_reason"] = neg_reason

        # Hallucination
        hal_score, hal_clean, hal_items = score_hallucination(input_words, raw_text)
        cs.hallucination_score = hal_score
        cs.no_medical_hallucination = hal_clean
        if not hal_clean:
            med_items = [h for h in hal_items if h["category"] != "adverb"]
            adv_items = [h for h in hal_items if h["category"] == "adverb"]
            if med_items:
                cs.failure_types.append(FailureType.HALLUCINATION_MEDICAL)
            if adv_items:
                cs.failure_types.append(FailureType.HALLUCINATION_ADVERB)
            cs.details["hallucinations"] = hal_items

        # Length
        len_score, len_failures = score_length(raw_text)
        cs.length_score = len_score
        cs.failure_types.extend(len_failures)

        # Total weighted score
        cfg = self.config
        cs.total_score = (
            cfg.w_keyword * cs.keyword_score
            + cfg.w_negation * cs.negation_score
            + cfg.w_hallucination * cs.hallucination_score
            + cfg.w_length * cs.length_score
        )

        return cs

    def score_all(
        self, input_words: list[str], raw_candidates: list[str],
    ) -> list[CandidateScore]:
        """Score all raw candidates."""
        return [self.score_candidate(input_words, c) for c in raw_candidates]
