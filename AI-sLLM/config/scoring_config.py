"""Constraint scoring configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ScoringConfig:
    """Controls Layer 2: Constraint Scoring and Layer 3: Selection."""

    # Score weights (must sum to 1.0)
    w_keyword: float = 0.35
    w_negation: float = 0.25
    w_hallucination: float = 0.25
    w_length: float = 0.15

    # Selection thresholds
    min_acceptable_score: float = 0.4
    perfect_score_threshold: float = 0.95

    # Safety flags for fallback ranking
    critical_keyword_weight: float = 2.0  # boost for candidates preserving all keywords
    negation_safety_weight: float = 2.0   # boost for negation-consistent candidates
