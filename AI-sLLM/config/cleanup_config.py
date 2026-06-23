"""Cleanup layer configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CleanupConfig:
    """Controls Layer 4: Format Cleanup and Layer 5: Semantic Cleanup."""

    # Format cleanup (Layer 4) - always on
    enable_prefix_removal: bool = True
    enable_sentence_clipping: bool = True
    enable_spacing_fix: bool = True
    enable_punctuation_fix: bool = True
    enable_plain_to_polite: bool = True
    enable_token_repair: bool = True

    # Semantic cleanup (Layer 5) - individually toggleable
    enable_english_replacement: bool = True
    enable_tense_rewrite: bool = True
    enable_grammar_fix: bool = True
    enable_hallucination_removal: bool = True

    # Tracing
    enable_tracing: bool = False
    trace_dir: str = "traces"
