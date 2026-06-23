"""Inference trace recorder.

Records full pipeline execution for debugging and analysis.
Each inference produces a JSON trace file when tracing is enabled.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class InferenceTrace:
    """Complete trace of one inference run."""

    # Input
    request_id: str = ""
    input_words: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    # Layer 1: Raw Generation
    raw_candidates: list[str] = field(default_factory=list)
    generation_method: list[str] = field(default_factory=list)  # ["greedy", "sampling", "sampling"]

    # Layer 2: Constraint Scoring
    scores: list[dict] = field(default_factory=list)

    # Layer 3: Candidate Selection
    selected_index: int = -1
    fallback_index: int = -1
    selection_reason: str = ""

    # Layer 4 + 5: Cleanup
    best_after_format_cleanup: str = ""
    best_after_semantic_cleanup: str = ""
    fallback_after_format_cleanup: str = ""
    fallback_after_semantic_cleanup: str = ""
    cleanup_changes: list[dict] = field(default_factory=list)

    # Layer 6: Guardrail
    guardrail_passed: bool = False
    used_fallback: bool = False
    guardrail_warnings: list[str] = field(default_factory=list)

    # Final output
    final_output: str = ""
    failure_types: list[str] = field(default_factory=list)

    # Timing
    duration_ms: float = 0.0


class TraceRecorder:
    """Records and saves inference traces."""

    def __init__(self, trace_dir: str = "traces", enabled: bool = False):
        self.enabled = enabled
        self.trace_dir = Path(trace_dir)
        if enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: InferenceTrace) -> str | None:
        """Save trace to JSON file. Returns file path or None if disabled."""
        if not self.enabled:
            return None

        ts = int(trace.timestamp * 1000)
        rid = trace.request_id or "unknown"
        filename = f"{ts}_{rid}.json"
        filepath = self.trace_dir / filename

        data = asdict(trace)
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(filepath)
