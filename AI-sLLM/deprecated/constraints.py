"""Lightweight constraint system for medical information preservation."""

from __future__ import annotations

import re
from typing import Callable

from medical_vocab import (
    SYMPTOM_KEYWORDS,
    BODY_PARTS,
    TEST_KEYWORDS,
    MEDICATION_KEYWORDS,
    NEGATION_WORDS,
    HALLUCINATION_ENTITIES,
)

# ---------------------------------------------------------------------------
# Checking functions
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,.!?。]+", text) if t]


def extract_input_keywords(words: list[str]) -> set[str]:
    """Return set of content words from input (excluding purely grammatical)."""
    function_words = {"나", "너", "우리", "그", "이", "저", "누구", "무엇",
                      "언제", "어디", "어떻게", "왜", "안", "못", "네", "예"}
    return {w for w in words if w not in function_words}


def check_keyword_preservation(
    input_words: list[str], prediction: str
) -> tuple[bool, list[str]]:
    """Check if all input content words appear in the prediction.

    Uses lightweight morphology: checks stems for verb/adjective conjugation
    (e.g., 가다→간, 아프다→아파) without full kiwi analysis.

    Returns (pass, missing_keywords).
    """
    keywords = extract_input_keywords(input_words)
    pred_lower = prediction.lower()
    pred_nospace = pred_lower.replace(" ", "")

    missing: list[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        # 1. Direct substring match
        if kw_lower in pred_lower:
            continue
        # 2. Stem match for verbs/adjectives ending in 다
        if kw.endswith("다") and len(kw) > 1:
            stem = kw[:-1]
            if stem in pred_lower:
                continue
        # 3. Spacing variation (못자다 → 못 자)
        kw_nospace = kw_lower.replace(" ", "")
        if len(kw_nospace) >= 2 and kw_nospace in pred_nospace:
            continue
        missing.append(kw)

    return (len(missing) == 0), missing


def check_negation_preservation(
    input_words: list[str], prediction: str
) -> tuple[bool, str]:
    """Check that negation in input is preserved (not reversed).

    Returns (pass, reason).
    """
    has_negation = any(nw in " ".join(input_words) for nw in NEGATION_WORDS)
    if not has_negation:
        return True, ""

    # Check if prediction flipped the meaning
    pred_lower = prediction.lower()

    # Look for explicit affirmation in prediction when input has negation
    input_joined = " ".join(input_words)
    has_neg = any(nw in input_joined for nw in NEGATION_WORDS)
    has_pos = any(pw in pred_lower for pw in ["있다", "있음", "가능", "할 수 있다"])

    # If input has "없다" but output has "있다" → negation reversal
    neg_word_found = None
    for nw in NEGATION_WORDS:
        if nw in input_joined:
            neg_word_found = nw
            break

    if neg_word_found and has_pos:
        # Check specific: if "없다"/"없음" in input, "있다" in output is reversal
        if ("없다" in input_joined or "없음" in input_joined) and "있다" in pred_lower:
            return False, f"negation_reversed: input has '{neg_word_found}' but output has affirmative"

        if ("못" in input_joined or "안" in input_joined) and ("가능" in pred_lower or "할 수" in pred_lower):
            return False, f"negation_reversed: input has '{neg_word_found}' but output has positive"

    # Check that the negation word itself appears
    if neg_word_found and neg_word_found not in pred_lower:
        # Some mappings:  못하다 → 못하, 필요없다 → 필요가 없다
        # Relax: check if the root morpheme appears
        root = neg_word_found.replace(" ", "").replace("없다", "없").replace("못하다", "못")
        if root not in pred_lower.replace(" ", ""):
            return False, f"negation_missing: input has '{neg_word_found}' but output doesn't contain it"

    return True, ""



# Common Korean particles/postpositions that follow nouns.
# If the character after a match is one of these, the match is still independent.
# e.g., 간+이, 위+가, 약+을, 열+이 → the entity is independent, the suffix is a particle.
_PARTICLES = set("이가을를은는도에서의와과로만부터까지한")


def _is_independent_word(word: str, text: str) -> bool:
    """Check if word appears as an independent unit in text (not as a substring).

    Lightweight morphology: uses Korean character boundary + particle awareness
    instead of full kiwi analysis. Suitable for runtime constraint checks.

    Examples:
      - '간' in '간다'        → False (conjugation of 가다)
      - '간' in '간이 아프다'  → True  (간+이 = liver + particle)
      - '간' in '간호사'       → False (substring of 간호사)
      - '위' in '위해'         → False (part of 위해)
      - '약' in '예약'         → False (substring of 예약)
      - '약' in '약을 먹는다'  → True  (약+을 = medicine + particle)
    """
    for match in re.finditer(re.escape(word), text):
        start, end = match.start(), match.end()
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "

        # Korean syllable range: 가(0xAC00) ~ 힣(0xD7A3)
        before_is_korean = "가" <= before <= "힣"
        after_is_korean = "가" <= after <= "힣"

        # Before: must not be part of a larger word
        if before_is_korean:
            continue

        # After: either not Korean, or is a known particle
        if after_is_korean and after not in _PARTICLES:
            continue

        return True
    return False


def check_hallucination(
    input_words: list[str], prediction: str
) -> tuple[bool, list[dict[str, str]]]:
    """Detect if prediction contains medical entities not present in input.

    Uses lightweight morphology: checks if clinical terms appear as independent
    words (not substrings of larger words) to avoid false positives like
    간 in 간다, 위 in 위해, 약 in 예약.

    Returns (pass, list of hallucinated entities).
    """
    input_text = " ".join(input_words).lower()
    pred_text = prediction.lower()

    # Also check input stems (아프다 → 아프)
    input_stems = set(input_words)
    for w in input_words:
        if w.endswith("다") and len(w) > 1:
            input_stems.add(w[:-1])

    hallucinations: list[dict[str, str]] = []

    for category, entities in HALLUCINATION_ENTITIES.items():
        for entity in entities:
            e = entity.strip().lower()
            if not e or e in input_text or e in input_stems:
                continue
            # Use lightweight boundary check instead of naive `in`
            if _is_independent_word(e, pred_text):
                hallucinations.append({
                    "category": category,
                    "entity": entity.strip(),
                })

    return (len(hallucinations) == 0), hallucinations


# ---------------------------------------------------------------------------
# Main ConstraintSystem
# ---------------------------------------------------------------------------


class ConstraintSystem:
    """Lightweight constraint system with configurable checks and retry."""

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries

    def check_all(
        self, input_words: list[str], prediction: str
    ) -> dict:
        """Run all constraint checks on a prediction.

        Returns a report dict.
        """
        kw_ok, kw_missing = check_keyword_preservation(input_words, prediction)
        neg_ok, neg_reason = check_negation_preservation(input_words, prediction)
        hal_ok, hal_items = check_hallucination(input_words, prediction)

        violations: list[str] = []
        if not kw_ok:
            violations.append("keyword_missing")
        if not neg_ok:
            violations.append("negation_error")
        if not hal_ok:
            violations.append("hallucination")

        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "keyword_missing": kw_missing,
            "negation_reason": neg_reason,
            "hallucinations": hal_items,
        }

    def generate_with_constraints(
        self,
        words: list[str],
        generator_fn: Callable[[list[str]], str],
        retry_fn: Callable[[list[str]], str] | None = None,
    ) -> str:
        """Generate with retry on constraint violations.

        The generator_fn should take `words` and return a sentence string.
        retry_fn is an optional alternative generator used on retries (e.g.
        with sampling enabled) so retries can produce different output.
        If not provided and do_sample=False, retries are skipped since
        greedy decoding always produces the same result.
        """
        prediction = generator_fn(words)
        report = self.check_all(words, prediction)

        if report["passed"] or retry_fn is None:
            return prediction

        # Retry with sampling-based generator for different results
        best = prediction
        for _ in range(self.max_retries):
            candidate = retry_fn(words)
            report = self.check_all(words, candidate)
            best = candidate
            if report["passed"]:
                break

        return best

    @staticmethod
    def format_report(report: dict) -> str:
        if report["passed"]:
            return "✅ All constraints passed"
        parts = []
        if "keyword_missing" in report["violations"]:
            parts.append(f"❗ Missing keywords: {report['keyword_missing']}")
        if "negation_error" in report["violations"]:
            parts.append(f"❗ Negation: {report['negation_reason']}")
        if "hallucination" in report["violations"]:
            ents = [h["entity"] for h in report["hallucinations"]]
            parts.append(f"❗ Hallucinated: {ents}")
        return "\n".join(parts)
