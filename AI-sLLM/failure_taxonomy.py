"""Failure type definitions for the inference pipeline.

Centralized taxonomy for classifying generation failures.
Used by constraint_scorer, candidate_selector, and inference_trace.
"""

from __future__ import annotations

from enum import Enum


class FailureType(Enum):
    """Enumeration of all recognized failure types."""

    # --- Constraint failures (Layer 2) ---
    KEYWORD_OMISSION = "keyword_omission"
    NEGATION_FLIP = "negation_flip"
    HALLUCINATION_MEDICAL = "hallucination_med"
    HALLUCINATION_ADVERB = "hallucination_adv"
    OVER_GENERATION = "over_generation"

    # --- Cleanup side-effects (Layer 5) ---
    TENSE_ERROR = "tense_error"
    GRAMMAR_ERROR = "grammar_error"
    SPACING_ERROR = "spacing_error"
    ENGLISH_LEAK = "english_leak"

    # --- Guardrail failures (Layer 6) ---
    EMPTY_OUTPUT = "empty_output"
    PROMPT_RESIDUE = "prompt_residue"
    OUTPUT_TOO_LONG = "output_too_long"
    FINAL_KEYWORD_MISSING = "final_kw_missing"
    FINAL_NEGATION_INCONSISTENT = "final_neg_inconsistent"
