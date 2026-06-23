"""Layer 3: Candidate Selection.

Selects best and fallback candidates from scored list.
Does NOT score — only selects. Scoring is in constraint_scorer.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config.scoring_config import ScoringConfig
from constraint_scorer import CandidateScore
from failure_taxonomy import FailureType


@dataclass
class SelectionResult:
    """Result of candidate selection."""

    best: CandidateScore
    fallback: CandidateScore | None = None
    all_scored: list[CandidateScore] = field(default_factory=list)
    failure_log: list[dict] = field(default_factory=list)
    selection_reason: str = ""


class CandidateSelector:
    """Selects best candidate using safety-aware ranking."""

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()

    def _safety_adjusted_score(self, cs: CandidateScore) -> float:
        """Compute safety-adjusted score for ranking.

        Beyond total_score, boosts candidates that are safe:
        - All keywords preserved
        - Negation consistent
        - No medical hallucination
        """
        cfg = self.config
        score = cs.total_score

        if cs.all_keywords_preserved:
            score += 0.1 * cfg.critical_keyword_weight
        if cs.negation_consistent:
            score += 0.1 * cfg.negation_safety_weight
        if cs.no_medical_hallucination:
            score += 0.05

        return score

    def _build_failure_log(self, scored: list[CandidateScore]) -> list[dict]:
        """Build failure log from all candidates."""
        log = []
        for i, cs in enumerate(scored):
            if cs.failure_types:
                log.append({
                    "candidate_index": i,
                    "raw_text": cs.raw_text[:80],
                    "total_score": round(cs.total_score, 3),
                    "failure_types": [f.value for f in cs.failure_types],
                    "details": cs.details,
                })
        return log

    def select(self, scored_candidates: list[CandidateScore]) -> SelectionResult:
        """Select best and fallback from scored candidates.

        Ranking uses safety-adjusted score, not just total_score.
        This ensures fallback is the safest alternative, not just
        the most fluent one.
        """
        if not scored_candidates:
            raise ValueError("No candidates to select from")

        # Rank by safety-adjusted score
        ranked = sorted(
            scored_candidates,
            key=lambda cs: self._safety_adjusted_score(cs),
            reverse=True,
        )

        best = ranked[0]
        fallback = ranked[1] if len(ranked) > 1 else None

        # Determine selection reason
        if best.total_score >= self.config.perfect_score_threshold:
            reason = "perfect_score"
        elif best.all_keywords_preserved and best.negation_consistent:
            reason = "safe_candidate"
        elif not best.failure_types:
            reason = "no_failures"
        else:
            reason = f"best_available (score={best.total_score:.3f})"

        return SelectionResult(
            best=best,
            fallback=fallback,
            all_scored=list(scored_candidates),
            failure_log=self._build_failure_log(scored_candidates),
            selection_reason=reason,
        )
