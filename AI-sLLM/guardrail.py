"""Layer 6: Final Guardrail.

Last line of defense before output.
Checks both format validity and semantic safety.
Falls back to pre-cleaned fallback candidate if best fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from failure_taxonomy import FailureType


@dataclass
class GuardrailResult:
    """Result of final guardrail check."""

    output: str
    passed: bool
    used_fallback: bool = False
    failure_types: list[FailureType] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FinalGuardrail:
    """Final validation combining format checks and semantic safety."""

    def __init__(self, max_length: int = 120):
        self.max_length = max_length

    def check(
        self,
        best_cleaned: str,
        fallback_cleaned: str | None,
        input_words: list[str],
    ) -> GuardrailResult:
        """Run final guardrail on cleaned best candidate.

        If best fails critical checks, switches to fallback.
        If both fail, returns best with warnings.
        """
        # Try best first
        best_result = self._validate(best_cleaned, input_words)
        if best_result.passed:
            return best_result

        # Best failed — try fallback
        if fallback_cleaned:
            fallback_result = self._validate(fallback_cleaned, input_words)
            if fallback_result.passed:
                fallback_result.used_fallback = True
                fallback_result.warnings.append(
                    f"best failed: {[f.value for f in best_result.failure_types]}"
                )
                return fallback_result

        # Both failed — return best with warning
        best_result.warnings.append("both best and fallback failed guardrail")
        return best_result

    def _validate(self, text: str, input_words: list[str]) -> GuardrailResult:
        """Validate a single candidate."""
        failures: list[FailureType] = []

        # --- Format checks ---

        # Empty
        if not text or not text.strip():
            return GuardrailResult(
                output=text, passed=False,
                failure_types=[FailureType.EMPTY_OUTPUT],
            )

        # Too long
        if len(text) > self.max_length:
            failures.append(FailureType.OUTPUT_TOO_LONG)

        # Prompt residue
        if any(marker in text for marker in ["입력", "출력", "예시", "규칙", "단어:"]):
            failures.append(FailureType.PROMPT_RESIDUE)

        # --- Semantic safety checks ---

        # Final keyword preservation
        content_words = {w for w in input_words
                         if w not in {"나", "너", "우리", "그", "이", "저",
                                      "누구", "무엇", "언제", "어디", "어떻게",
                                      "왜", "안", "못", "네", "예"}}
        text_lower = text.lower()
        for kw in content_words:
            kw_lower = kw.lower()
            found = False
            # Direct match
            if kw_lower in text_lower:
                found = True
            # Stem match
            elif kw.endswith("다") and len(kw) > 1 and kw[:-1] in text_lower:
                found = True
            # Nospace match
            elif kw_lower.replace(" ", "") in text_lower.replace(" ", ""):
                found = True
            if not found:
                failures.append(FailureType.FINAL_KEYWORD_MISSING)
                break  # One missing keyword is enough to flag

        # Negation consistency
        from medical_vocab import NEGATION_WORDS
        input_joined = " ".join(input_words)
        has_negation = any(nw in input_joined for nw in NEGATION_WORDS)
        if has_negation:
            # Check if negation is preserved
            neg_found = False
            for nw in NEGATION_WORDS:
                if nw in input_joined:
                    root = nw.replace("없다", "없").replace("못하다", "못").replace(" ", "")
                    if root in text_lower.replace(" ", ""):
                        neg_found = True
                        break
            if not neg_found:
                failures.append(FailureType.FINAL_NEGATION_INCONSISTENT)

        passed = len(failures) == 0
        return GuardrailResult(output=text, passed=passed, failure_types=failures)
