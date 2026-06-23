"""Generation layer configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Controls Layer 1: Raw Generation."""

    # Candidate generation
    num_sampling_candidates: int = 2  # + 1 greedy = 3 total
    sampling_temperature: float = 0.7
    sampling_top_p: float = 0.9

    # Model inference
    max_new_tokens_base: int = 48     # EXAONE base
    max_new_tokens_finetuned: int = 64  # Fine-tuned
    repetition_penalty: float = 1.3

    # Prompt
    prompt_type: str = "fewshot"  # "basic" or "fewshot"
