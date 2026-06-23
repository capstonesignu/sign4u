"""Error Analysis Tool for AI-sLLM Pipeline.

Reads trace files and evaluation results to produce a systematic
error analysis report. Classifies errors into:
  - Model-level issues (reasoning, narrative expansion, syntax)
  - Pipeline-level issues (cleanup side-effects, scoring gaps)

Usage:
    python scripts/error_analysis.py [--traces traces_v2] [--results human_eval_v2_results.json]
"""

from __future__ import annotations

import json
import re
import sys
import io
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Error categories (beyond FailureType — higher-level analysis)
# ---------------------------------------------------------------------------

MODEL_ERRORS = {
    "causal_reasoning":   "因果推理错误 (Causal Reasoning)",
    "narrative_expansion": "叙述扩展 (Narrative Expansion)",
    "syntax_composition": "句法组合不稳定 (Syntax Composition)",
    "english_leak":       "英语泄漏 (English Leak)",
    "negation_reversal":  "否定反转 (Negation Reversal)",
}

CLEANUP_ERRORS = {
    "tense_side_effect":   "时态修改副作用 (Tense Side-Effect)",
    "english_replace_err": "英韩替换错误 (English Replace Error)",
    "grammar_side_effect": "语法修正副作用 (Grammar Side-Effect)",
    "halluc_remove_err":   "幻觉删除过度 (Hallucination Removal Error)",
    "cleanup_rollback":    "Cleanup 回退触发 (Rollback Triggered)",
}

# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

# Narrative expansion markers: model adds verbs not in input
EXPANSION_VERBS = {
    "말한다", "확인한다", "느낀다", "한다고", "보인다", "알았다",
    "했다고", "한다는", "듣는다", "본다", "이야기", "호소",
}

# English word pattern (excluding known allowed abbreviations)
ENGLISH_WHITELIST = {"CT", "MRI", "X-ray", "X", "PET", "ICU", "ER", "IV", "OK"}


@dataclass
class SampleAnalysis:
    """Analysis result for one sample."""
    sample_id: str = ""
    input_words: list[str] = field(default_factory=list)
    reference: str = ""
    system_output: str = ""
    is_exact_match: bool = False
    error_categories: list[str] = field(default_factory=list)
    error_details: list[str] = field(default_factory=list)
    error_source: str = ""  # "model" or "cleanup" or "both" or "none"

    # From trace (if available)
    trace_failures: list[str] = field(default_factory=list)
    cleanup_changes: list[dict] = field(default_factory=list)
    used_fallback: bool = False


def detect_english_leak(output: str) -> bool:
    """Detect English words in output (excluding whitelist)."""
    words = re.findall(r"[a-zA-Z]{2,}", output)
    for w in words:
        if w.upper() not in ENGLISH_WHITELIST and w not in ENGLISH_WHITELIST:
            return True
    return False


def detect_narrative_expansion(output: str, input_words: list[str]) -> list[str]:
    """Detect added narrative verbs not in input."""
    found = []
    for verb in EXPANSION_VERBS:
        if verb in output:
            # Check if the root is in input
            root = verb.rstrip("다는고")
            if not any(root in w for w in input_words):
                found.append(verb)
    return found


def detect_causal_error(output: str, reference: str, input_words: list[str]) -> bool:
    """Detect causal direction reversal (e.g., drinking water → thirsty)."""
    if not reference:
        return False
    # Simple heuristic: word order in output reverses causal direction from reference
    # Check if the first content word in output vs reference differs significantly
    out_words = output.replace(".", "").replace("?", "").replace("!", "").split()
    ref_words = reference.replace(".", "").replace("?", "").replace("!", "").split()
    if len(out_words) < 2 or len(ref_words) < 2:
        return False

    # Compare if input word order is reversed in output vs reference
    input_positions_out = []
    input_positions_ref = []
    for iw in input_words:
        stem = iw.rstrip("다") if iw.endswith("다") and len(iw) > 1 else iw
        for j, ow in enumerate(out_words):
            if stem in ow:
                input_positions_out.append(j)
                break
        for j, rw in enumerate(ref_words):
            if stem in rw:
                input_positions_ref.append(j)
                break

    if len(input_positions_out) >= 2 and len(input_positions_ref) >= 2:
        # Check if order is reversed
        out_order = input_positions_out[0] < input_positions_out[-1]
        ref_order = input_positions_ref[0] < input_positions_ref[-1]
        if out_order != ref_order:
            return True
    return False


def detect_syntax_issues(output: str) -> list[str]:
    """Detect Korean syntax composition problems."""
    issues = []
    # Missing spaces between particles and nouns
    if re.search(r"[가-힣]{6,}", output):
        # Very long compound without spaces might indicate spacing issue
        long_chunks = re.findall(r"[가-힣]{6,}", output)
        for chunk in long_chunks:
            # Check if it looks like a spacing error (not a legitimate compound)
            if not any(chunk in w for w in ["물리치료", "진통제", "처방전", "초음파", "혈액검사"]):
                issues.append(f"spacing: {chunk}")

    # Unnatural particle combinations
    if "을를" in output or "이가" in output or "은는" in output:
        issues.append("double_particle")

    return issues


def analyze_sample(
    sample: dict,
    trace: dict | None = None,
) -> SampleAnalysis:
    """Analyze a single sample for error categories."""
    analysis = SampleAnalysis(
        sample_id=sample.get("id", ""),
        input_words=sample.get("words", []),
        reference=sample.get("reference", ""),
        system_output=sample.get("system_output", ""),
    )

    ref = analysis.reference.strip()
    out = analysis.system_output.strip()
    words = analysis.input_words

    # Exact match check
    analysis.is_exact_match = (out == ref)
    if analysis.is_exact_match:
        analysis.error_source = "none"
        return analysis

    # Load trace data if available
    if trace:
        analysis.trace_failures = trace.get("failure_types", [])
        analysis.cleanup_changes = trace.get("cleanup_changes", [])
        analysis.used_fallback = trace.get("used_fallback", False)

    # --- Detect model-level errors ---
    model_errors = []

    # 1. English leak
    if detect_english_leak(out):
        model_errors.append("english_leak")
        analysis.error_details.append(
            f"English words in output: {re.findall(r'[a-zA-Z]{2,}', out)}"
        )

    # 2. Narrative expansion
    expansions = detect_narrative_expansion(out, words)
    if expansions:
        model_errors.append("narrative_expansion")
        analysis.error_details.append(f"Added verbs: {expansions}")

    # 3. Causal reasoning
    if detect_causal_error(out, ref, words):
        model_errors.append("causal_reasoning")
        analysis.error_details.append("Word order suggests causal reversal")

    # 4. Syntax composition
    syntax = detect_syntax_issues(out)
    if syntax:
        model_errors.append("syntax_composition")
        analysis.error_details.append(f"Syntax issues: {syntax}")

    # --- Detect cleanup side-effects ---
    cleanup_errors = []
    if trace:
        for change in analysis.cleanup_changes:
            ctype = change.get("type", "")
            if ctype == "rollback":
                cleanup_errors.append("cleanup_rollback")
                analysis.error_details.append(f"Rollback: {change.get('before', '')}")
            elif ctype == "tense_rewrite":
                # Check if tense rewrite caused issues
                if ref and change.get("after", "") not in ref:
                    cleanup_errors.append("tense_side_effect")
            elif ctype == "english_replace":
                if ref and change.get("after", "") not in ref:
                    cleanup_errors.append("english_replace_err")

    analysis.error_categories = model_errors + cleanup_errors

    # Determine source
    if model_errors and cleanup_errors:
        analysis.error_source = "both"
    elif model_errors:
        analysis.error_source = "model"
    elif cleanup_errors:
        analysis.error_source = "cleanup"
    else:
        # Not exact match but no detected pattern — likely subtle model issue
        analysis.error_source = "model"
        analysis.error_categories.append("unclassified_model")

    return analysis


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    analyses: list[SampleAnalysis],
    sample_categories: dict[str, str] | None = None,
) -> str:
    """Generate a comprehensive error analysis report."""
    lines: list[str] = []

    total = len(analyses)
    exact = sum(1 for a in analyses if a.is_exact_match)
    errors = [a for a in analyses if not a.is_exact_match]

    lines.append("=" * 65)
    lines.append("AI-sLLM Error Analysis Report")
    lines.append("=" * 65)
    lines.append("")

    # --- Summary ---
    lines.append(f"총 샘플: {total}")
    lines.append(f"완전 일치: {exact}/{total} ({exact/total*100:.1f}%)")
    lines.append(f"오류 샘플: {len(errors)}/{total} ({len(errors)/total*100:.1f}%)")
    lines.append("")

    # --- Error source breakdown ---
    source_counts = Counter(a.error_source for a in errors)
    lines.append("-" * 65)
    lines.append("오류 원인 분류 (Error Source Breakdown)")
    lines.append("-" * 65)
    lines.append(f"  모델 원인 (Model):     {source_counts.get('model', 0)} ({source_counts.get('model', 0)/max(len(errors),1)*100:.1f}%)")
    lines.append(f"  파이프라인 (Cleanup):   {source_counts.get('cleanup', 0)} ({source_counts.get('cleanup', 0)/max(len(errors),1)*100:.1f}%)")
    lines.append(f"  복합 원인 (Both):      {source_counts.get('both', 0)} ({source_counts.get('both', 0)/max(len(errors),1)*100:.1f}%)")
    lines.append("")

    # --- Error category frequency ---
    cat_counts = Counter()
    for a in errors:
        for cat in a.error_categories:
            cat_counts[cat] += 1

    lines.append("-" * 65)
    lines.append("오류 유형별 빈도 (Error Category Frequency)")
    lines.append("-" * 65)

    # Model errors
    lines.append("  [모델 오류 / Model Errors]")
    for key, label in MODEL_ERRORS.items():
        count = cat_counts.get(key, 0)
        if count > 0:
            lines.append(f"    {label}: {count} ({count/max(len(errors),1)*100:.1f}%)")
    unclassified = cat_counts.get("unclassified_model", 0)
    if unclassified:
        lines.append(f"    미분류 모델 오류 (Unclassified): {unclassified} ({unclassified/max(len(errors),1)*100:.1f}%)")

    lines.append("")
    lines.append("  [파이프라인 오류 / Cleanup Errors]")
    for key, label in CLEANUP_ERRORS.items():
        count = cat_counts.get(key, 0)
        if count > 0:
            lines.append(f"    {label}: {count} ({count/max(len(errors),1)*100:.1f}%)")
    lines.append("")

    # --- Category breakdown (daily vs medical) ---
    if sample_categories:
        lines.append("-" * 65)
        lines.append("카테고리별 오류 분포 (By Sample Category)")
        lines.append("-" * 65)

        cat_groups: dict[str, list[SampleAnalysis]] = defaultdict(list)
        for a in analyses:
            cat = sample_categories.get(a.sample_id, "unknown")
            cat_groups[cat].append(a)

        for cat_name in sorted(cat_groups.keys()):
            group = cat_groups[cat_name]
            group_total = len(group)
            group_exact = sum(1 for a in group if a.is_exact_match)
            group_errors = [a for a in group if not a.is_exact_match]

            lines.append(f"  [{cat_name}] {group_exact}/{group_total} 일치 ({group_exact/max(group_total,1)*100:.1f}%)")

            group_cats = Counter()
            for a in group_errors:
                for c in a.error_categories:
                    group_cats[c] += 1

            for c, cnt in group_cats.most_common(5):
                label = MODEL_ERRORS.get(c, CLEANUP_ERRORS.get(c, c))
                lines.append(f"    {label}: {cnt}")
            lines.append("")

    # --- Trace-based failure type stats ---
    trace_failures = Counter()
    for a in analyses:
        for f in a.trace_failures:
            trace_failures[f] += 1

    if trace_failures:
        lines.append("-" * 65)
        lines.append("파이프라인 Failure Type 통계 (Pipeline FailureType Stats)")
        lines.append("-" * 65)
        for ft, count in trace_failures.most_common():
            lines.append(f"  {ft}: {count}")
        lines.append("")

    # --- Cleanup change stats ---
    change_types = Counter()
    for a in analyses:
        for ch in a.cleanup_changes:
            change_types[ch.get("type", "unknown")] += 1

    if change_types:
        lines.append("-" * 65)
        lines.append("Semantic Cleanup 변경 통계 (Cleanup Change Stats)")
        lines.append("-" * 65)
        for ct, count in change_types.most_common():
            lines.append(f"  {ct}: {count}")
        lines.append("")

    # --- Worst samples (most error categories) ---
    lines.append("-" * 65)
    lines.append("주요 오류 샘플 (Top Error Samples)")
    lines.append("-" * 65)

    sorted_errors = sorted(errors, key=lambda a: len(a.error_categories), reverse=True)
    for a in sorted_errors[:15]:
        cats_str = ", ".join(a.error_categories) if a.error_categories else "unclassified"
        lines.append(f"  [{a.sample_id}] ({a.error_source}) {cats_str}")
        lines.append(f"    입력: {' / '.join(a.input_words)}")
        lines.append(f"    참고: {a.reference}")
        lines.append(f"    출력: {a.system_output}")
        if a.error_details:
            for detail in a.error_details[:3]:
                lines.append(f"    → {detail}")
        lines.append("")

    # --- Conclusions ---
    lines.append("=" * 65)
    lines.append("결론 (Conclusions)")
    lines.append("=" * 65)

    model_total = source_counts.get("model", 0) + source_counts.get("both", 0)
    cleanup_total = source_counts.get("cleanup", 0) + source_counts.get("both", 0)

    lines.append(f"  모델 관련 오류: {model_total}/{len(errors)} ({model_total/max(len(errors),1)*100:.1f}%)")
    lines.append(f"  파이프라인 관련 오류: {cleanup_total}/{len(errors)} ({cleanup_total/max(len(errors),1)*100:.1f}%)")
    lines.append("")

    if cat_counts.most_common(1):
        top_error, top_count = cat_counts.most_common(1)[0]
        top_label = MODEL_ERRORS.get(top_error, CLEANUP_ERRORS.get(top_error, top_error))
        lines.append(f"  가장 빈번한 오류: {top_label} ({top_count}건)")

    lines.append("")
    lines.append("=" * 65)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_traces(trace_dir: Path) -> dict[str, dict]:
    """Load all trace files, keyed by request_id."""
    traces = {}
    if not trace_dir.exists():
        return traces

    for f in trace_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            rid = data.get("request_id", "")
            if rid:
                traces[rid] = data
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return traces


def classify_samples(results: list[dict]) -> dict[str, str]:
    """Classify samples as 'daily' or 'medical' based on ID ranges."""
    categories = {}
    for r in results:
        sid = r.get("id", "")
        # Samples 1-24 are daily, 25+ are medical (based on test_data_clean.json structure)
        try:
            num = int(sid.replace("sample-", ""))
            categories[sid] = "일상 (Daily)" if num <= 24 else "의료 (Medical)"
        except ValueError:
            categories[sid] = "unknown"
    return categories


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI-sLLM Error Analysis")
    parser.add_argument("--results", default="human_eval_v2_results.json",
                        help="Path to results JSON")
    parser.add_argument("--traces", default="traces_v2",
                        help="Path to trace directory")
    parser.add_argument("--output", default="error_analysis_report.txt",
                        help="Output report path")
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} not found")
        sys.exit(1)

    results = json.loads(results_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(results)} results from {results_path}")

    # Load traces
    traces = load_traces(Path(args.traces))
    print(f"Loaded {len(traces)} traces from {args.traces}")

    # Classify samples
    sample_categories = classify_samples(results)

    # Analyze each sample
    analyses = []
    for sample in results:
        sid = sample.get("id", "")
        trace = traces.get(sid)
        analysis = analyze_sample(sample, trace)
        analyses.append(analysis)

    # Generate report
    report = generate_report(analyses, sample_categories)
    print(report)

    # Save report
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
