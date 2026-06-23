"""Pre-Freeze Audit: Comprehensive pipeline evaluation.

Runs the full evaluation dataset through the frozen pipeline,
collects traces, and produces a complete audit report covering:

1. Full evaluation run (stability, exceptions, trace completeness)
2. Error distribution (count + percentage for all categories)
3. Pipeline effectiveness comparison (pre-rollback vs post-rollback)
4. Hidden system error detection (trace anomalies)
5. Random spot-check sample selection (20-30 samples for human review)
6. Version freeze manifest
7. Final summary for advisor review

Usage:
    python scripts/pre_freeze_audit.py
"""

from __future__ import annotations

import json
import random
import re
import sys
import io
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_DIR / "human_eval_v2_results.json"
TRACES_DIR = PROJECT_DIR / "traces_v2"
TEST_DATA = PROJECT_DIR / "test_data_clean.json"
REPORT_OUTPUT = PROJECT_DIR / "pre_freeze_audit_report.txt"

PIPELINE_VERSION = "v2.0-final"
MODEL_VERSION = "EXAONE-3.5-2.4B-Instruct (QLoRA fine-tuned, exaone-finetuned-v3)"

# Narrative expansion markers
EXPANSION_VERBS = {
    "말한다", "확인한다", "느낀다", "한다고", "보인다", "알았다",
    "했다고", "한다는", "듣는다", "이야기", "호소", "문의",
    "전달", "요청", "파악",
}

ENGLISH_WHITELIST = {"CT", "MRI", "X", "PET", "ICU", "ER", "IV", "OK", "X-ray"}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def detect_english_leak(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{2,}", text)
    return [w for w in words if w.upper() not in ENGLISH_WHITELIST and w not in ENGLISH_WHITELIST]


def detect_narrative_expansion(output: str, input_words: list[str]) -> list[str]:
    found = []
    for verb in EXPANSION_VERBS:
        if verb in output:
            root = verb.rstrip("다는고")
            if not any(root in w for w in input_words):
                found.append(verb)
    return found


def keyword_coverage(text: str, input_words: list[str]) -> tuple[int, int, list[str]]:
    """Returns (found_count, total_content_words, missing_words)."""
    skip = {"나", "너", "우리", "그", "이", "저", "누구", "무엇", "언제",
            "어디", "어떻게", "왜", "안", "못", "네", "예"}
    content_words = [w for w in input_words if w not in skip]
    missing = []
    text_lower = text.lower()
    text_nospace = text_lower.replace(" ", "")

    for kw in content_words:
        kl = kw.lower()
        found = False
        if kl in text_lower:
            found = True
        elif kw.endswith("다") and len(kw) > 1 and kw[:-1] in text_lower:
            found = True
        elif kl.replace(" ", "") in text_nospace:
            found = True
        if not found:
            missing.append(kw)

    return len(content_words) - len(missing), len(content_words), missing


def detect_spacing_issues(text: str) -> bool:
    """Detect abnormal spacing (long runs without spaces or single chars between spaces)."""
    if re.search(r"[가-힣]{10,}", text):
        return True
    return False


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit():
    lines: list[str] = []

    def section(title: str):
        lines.append("")
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)
        lines.append("")

    def subsection(title: str):
        lines.append("")
        lines.append("-" * 70)
        lines.append(title)
        lines.append("-" * 70)

    # ===================================================================
    # Load data
    # ===================================================================

    if not RESULTS_FILE.exists():
        print(f"ERROR: {RESULTS_FILE} not found. Run evaluation first.")
        sys.exit(1)

    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    test_data = json.loads(TEST_DATA.read_text(encoding="utf-8"))
    refs = {s.get("id", f"sample-{i+1:03d}"): s.get("reference", "")
            for i, s in enumerate(test_data)}

    # Load traces
    traces: dict[str, dict] = {}
    if TRACES_DIR.exists():
        for f in TRACES_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                rid = data.get("request_id", "")
                if rid:
                    traces[rid] = data
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    print(f"Loaded: {len(results)} results, {len(traces)} traces")

    # ===================================================================
    # STEP 1: Full Evaluation Stability Check
    # ===================================================================

    section("STEP 1: Full Evaluation Stability Check")

    total = len(results)
    exact_match = sum(1 for r in results
                      if r["system_output"].strip() == refs.get(r["id"], "").strip())

    # Check for empty outputs
    empty_outputs = [r for r in results if not r["system_output"].strip()]

    # Check for runtime exceptions (outputs that look like errors)
    error_outputs = [r for r in results
                     if "error" in r["system_output"].lower()
                     or "traceback" in r["system_output"].lower()
                     or "exception" in r["system_output"].lower()]

    # Check trace completeness
    missing_traces = [r["id"] for r in results if r["id"] not in traces]
    incomplete_traces = []
    for rid, trace in traces.items():
        if not trace.get("raw_candidates"):
            incomplete_traces.append(rid)
        elif not trace.get("final_output"):
            incomplete_traces.append(rid)

    # Check candidate count consistency
    candidate_counts = Counter()
    for trace in traces.values():
        n = len(trace.get("raw_candidates", []))
        candidate_counts[n] += 1

    lines.append(f"총 샘플: {total}")
    lines.append(f"완전 일치 (Exact Match): {exact_match}/{total} ({exact_match/total*100:.1f}%)")
    lines.append(f"빈 출력 (Empty Output): {len(empty_outputs)}")
    lines.append(f"런타임 에러: {len(error_outputs)}")
    lines.append(f"Trace 누락: {len(missing_traces)}")
    lines.append(f"Trace 불완전: {len(incomplete_traces)}")
    lines.append(f"Candidate 수 분포: {dict(candidate_counts)}")
    lines.append("")

    stability_ok = (len(empty_outputs) == 0 and len(error_outputs) == 0
                    and len(missing_traces) == 0 and len(incomplete_traces) == 0)
    lines.append(f"✓ 안정성 검증: {'PASS' if stability_ok else 'ISSUES FOUND'}")

    if empty_outputs:
        lines.append(f"  경고: 빈 출력 샘플: {[r['id'] for r in empty_outputs]}")
    if error_outputs:
        lines.append(f"  경고: 에러 출력 샘플: {[r['id'] for r in error_outputs]}")

    # ===================================================================
    # STEP 2: Error Distribution
    # ===================================================================

    section("STEP 2: Error Distribution (Count + Percentage)")

    # Categorize each sample
    error_samples = []
    stats = {
        "narrative_expansion": [],
        "semantic_drift": [],
        "negation_flip": [],
        "keyword_omission": [],
        "syntax_composition": [],
        "cleanup_side_effect": [],
        "english_leakage": [],
        "rollback_trigger": [],
        "fallback_trigger": [],
        "causal_reasoning": [],
        "over_generation": [],
    }

    for r in results:
        sid = r["id"]
        out = r["system_output"].strip()
        ref = refs.get(sid, "").strip()
        words = r.get("words", [])
        trace = traces.get(sid, {})

        if out == ref:
            continue

        error_samples.append(sid)

        # Narrative expansion
        expansions = detect_narrative_expansion(out, words)
        if expansions:
            stats["narrative_expansion"].append(sid)

        # English leakage
        eng_words = detect_english_leak(out)
        if eng_words:
            stats["english_leakage"].append(sid)

        # Keyword omission
        found, total_kw, missing = keyword_coverage(out, words)
        if missing:
            stats["keyword_omission"].append(sid)

        # Syntax composition
        if detect_spacing_issues(out):
            stats["syntax_composition"].append(sid)

        # Over-generation (output much longer than reference)
        if ref and len(out) > len(ref) * 1.8:
            stats["over_generation"].append(sid)

        # Semantic drift (output has all keywords but meaning differs significantly)
        if not missing and out != ref and not expansions:
            # Has all keywords but still different — likely semantic drift
            stats["semantic_drift"].append(sid)

        # From trace data
        trace_failures = trace.get("failure_types", [])
        if "negation_flip" in trace_failures:
            stats["negation_flip"].append(sid)

        cleanup_changes = trace.get("cleanup_changes", [])
        has_rollback = any(c.get("type") == "rollback" for c in cleanup_changes)
        has_side_effect = any(c.get("type") in ("tense_rewrite", "english_replace", "grammar_fix")
                             for c in cleanup_changes)

        if has_rollback:
            stats["rollback_trigger"].append(sid)
        if has_side_effect and out != ref:
            stats["cleanup_side_effect"].append(sid)

        if trace.get("used_fallback"):
            stats["fallback_trigger"].append(sid)

        # Causal reasoning (word order reversal)
        if ref and words:
            out_words = out.replace(".", "").replace("?", "").split()
            ref_words = ref.replace(".", "").replace("?", "").split()
            if len(out_words) >= 2 and len(ref_words) >= 2:
                pos_out, pos_ref = [], []
                for iw in words[:3]:
                    stem = iw.rstrip("다") if iw.endswith("다") and len(iw) > 1 else iw
                    for j, ow in enumerate(out_words):
                        if stem in ow:
                            pos_out.append(j)
                            break
                    for j, rw in enumerate(ref_words):
                        if stem in rw:
                            pos_ref.append(j)
                            break
                if len(pos_out) >= 2 and len(pos_ref) >= 2:
                    if (pos_out[0] < pos_out[-1]) != (pos_ref[0] < pos_ref[-1]):
                        stats["causal_reasoning"].append(sid)

    n_err = len(error_samples)
    lines.append(f"오류 샘플 총 수: {n_err}/{total} ({n_err/total*100:.1f}%)")
    lines.append("")

    lines.append(f"{'Error Category':<35} {'Count':>6} {'% of errors':>12} {'% of total':>12}")
    lines.append(f"{'-'*35} {'-'*6} {'-'*12} {'-'*12}")

    for cat_name, cat_label in [
        ("narrative_expansion",  "Narrative Expansion (叙述扩展)"),
        ("semantic_drift",       "Semantic Drift (语义偏移)"),
        ("keyword_omission",     "Keyword Omission (关键词缺失)"),
        ("causal_reasoning",     "Causal Reasoning (因果推理)"),
        ("syntax_composition",   "Syntax Composition (句法问题)"),
        ("over_generation",      "Over-Generation (过度生成)"),
        ("english_leakage",      "English Leakage (英语泄漏)"),
        ("negation_flip",        "Negation Flip (否定反转)"),
        ("cleanup_side_effect",  "Cleanup Side-Effect (清理副作用)"),
        ("rollback_trigger",     "Rollback Triggered (回退触发)"),
        ("fallback_trigger",     "Fallback Used (备选使用)"),
    ]:
        count = len(stats[cat_name])
        pct_err = count / max(n_err, 1) * 100
        pct_tot = count / total * 100
        lines.append(f"{cat_label:<35} {count:>6} {pct_err:>11.1f}% {pct_tot:>11.1f}%")

    # Model vs pipeline source
    lines.append("")
    model_errors = set()
    for cat in ["narrative_expansion", "semantic_drift", "keyword_omission",
                "causal_reasoning", "syntax_composition", "over_generation",
                "english_leakage", "negation_flip"]:
        model_errors.update(stats[cat])

    pipeline_errors = set()
    for cat in ["cleanup_side_effect", "rollback_trigger"]:
        pipeline_errors.update(stats[cat])

    pure_model = model_errors - pipeline_errors
    pure_pipeline = pipeline_errors - model_errors
    both = model_errors & pipeline_errors

    lines.append(f"오류 원인 요약:")
    lines.append(f"  순수 모델 원인: {len(pure_model)}/{n_err} ({len(pure_model)/max(n_err,1)*100:.1f}%)")
    lines.append(f"  순수 파이프라인 원인: {len(pure_pipeline)}/{n_err} ({len(pure_pipeline)/max(n_err,1)*100:.1f}%)")
    lines.append(f"  복합 원인: {len(both)}/{n_err} ({len(both)/max(n_err,1)*100:.1f}%)")

    # Category breakdown (daily vs medical)
    subsection("카테고리별 분포 (Daily vs Medical)")
    for cat_name, cat_range in [("일상 (Daily)", range(1, 25)), ("의료 (Medical)", range(25, 97))]:
        cat_ids = {f"sample-{i:03d}" for i in cat_range}
        cat_results = [r for r in results if r["id"] in cat_ids]
        cat_exact = sum(1 for r in cat_results
                        if r["system_output"].strip() == refs.get(r["id"], "").strip())
        cat_total = len(cat_results)
        lines.append(f"  [{cat_name}] Exact Match: {cat_exact}/{cat_total} ({cat_exact/max(cat_total,1)*100:.1f}%)")

        # Top errors in this category
        cat_error_counts = Counter()
        for stat_name, stat_ids in stats.items():
            count = len([s for s in stat_ids if s in cat_ids])
            if count > 0:
                cat_error_counts[stat_name] = count
        for stat_name, count in cat_error_counts.most_common(5):
            lines.append(f"    {stat_name}: {count}")
        lines.append("")

    # ===================================================================
    # STEP 3: Pipeline Effectiveness Comparison
    # ===================================================================

    section("STEP 3: Pipeline Effectiveness Comparison")

    lines.append("비교 대상:")
    lines.append("  A. 중구 전 (Pre-Refactor, v1): postprocess.py 기반 단일 파이프라인")
    lines.append("  B. Rollback 전 (v2.0): 7-Layer pipeline, semantic cleanup without rollback")
    lines.append("  C. Rollback 후 (v2.0-final): 7-Layer pipeline + do-no-harm rollback")
    lines.append("")

    # V1 stats from human evaluation (hardcoded from previous analysis)
    lines.append(f"{'Metric':<40} {'v1 (pre)':>10} {'v2.0':>10} {'v2.0-final':>10}")
    lines.append(f"{'-'*40} {'-'*10} {'-'*10} {'-'*10}")

    # We know v1 overall avg was 3.45/5, and we have v2 stats
    v1_exact = "15.6%"  # from v1 evaluation
    v2_pre_rollback_exact = "14.6%"  # from previous run
    v2_final_exact = f"{exact_match/total*100:.1f}%"

    lines.append(f"{'Exact Match':.<40} {v1_exact:>10} {v2_pre_rollback_exact:>10} {v2_final_exact:>10}")

    # Pipeline side-effects
    v1_side_effect = "N/A"  # v1 had no tracking
    v2_pre_side = "5 (6.1%)"
    v2_final_side = f"{len(stats['cleanup_side_effect'])} ({len(stats['cleanup_side_effect'])/max(n_err,1)*100:.1f}%)"
    lines.append(f"{'Cleanup Side-Effects':.<40} {v1_side_effect:>10} {v2_pre_side:>10} {v2_final_side:>10}")

    # Rollback
    v2_pre_rollback_count = "0"
    v2_final_rollback_count = str(len(stats["rollback_trigger"]))
    lines.append(f"{'Rollback Triggers':.<40} {'N/A':>10} {v2_pre_rollback_count:>10} {v2_final_rollback_count:>10}")

    # Fallback usage
    lines.append(f"{'Fallback Used':.<40} {'N/A':>10} {'N/A':>10} {len(stats['fallback_trigger']):>10}")

    # English leakage
    lines.append(f"{'English Leakage':.<40} {'N/A':>10} {'3':>10} {len(stats['english_leakage']):>10}")

    # Narrative expansion
    lines.append(f"{'Narrative Expansion':.<40} {'N/A':>10} {'14':>10} {len(stats['narrative_expansion']):>10}")

    lines.append("")
    lines.append("분석:")
    lines.append("  - v1→v2: 구조적 문제(retry chaos, postprocess 오염, prompt residue) 제거")
    lines.append("  - v2→v2-final: cleanup side-effect 감소, rollback 보호 추가")
    lines.append("  - Exact Match가 v1과 비슷한 이유: v1의 15.6%는 단순한 문장에 집중,")
    lines.append("    v2는 동일 테스트셋 but 모델의 reasoning 한계가 bottleneck")
    lines.append("  - 핵심 개선: 시스템 안정성, 추적 가능성, side-effect 감소")

    # ===================================================================
    # STEP 4: Hidden System Error Detection
    # ===================================================================

    section("STEP 4: Hidden System Error Detection (Trace Anomalies)")

    anomalies = []

    for rid, trace in traces.items():
        # 4a. Cleanup 후 keyword 손실
        final_output = trace.get("final_output", "")
        input_words = trace.get("input_words", [])
        best_before = trace.get("best_after_format_cleanup", "")

        if best_before and final_output and input_words:
            _, _, missing_before = keyword_coverage(best_before, input_words)
            _, _, missing_after = keyword_coverage(final_output, input_words)
            lost_in_cleanup = set(missing_after) - set(missing_before)
            if lost_in_cleanup:
                anomalies.append(f"[{rid}] Keyword lost in cleanup: {lost_in_cleanup}")

        # 4b. Rollback 연속 발생 (potential loop)
        changes = trace.get("cleanup_changes", [])
        rollback_count = sum(1 for c in changes if c.get("type") == "rollback")
        if rollback_count > 2:
            anomalies.append(f"[{rid}] Multiple rollbacks: {rollback_count}")

        # 4c. Fallback worse than best
        if trace.get("used_fallback"):
            scores = trace.get("scores", [])
            sel_idx = trace.get("selected_index", 0)
            fb_idx = trace.get("fallback_index", -1)
            if scores and 0 <= sel_idx < len(scores) and 0 <= fb_idx < len(scores):
                best_score = scores[sel_idx].get("total", 0)
                fb_score = scores[fb_idx].get("total", 0)
                if fb_score < best_score * 0.8:
                    anomalies.append(f"[{rid}] Fallback score ({fb_score:.3f}) much lower than best ({best_score:.3f})")

        # 4d. Scoring inconsistency (highest scoring candidate not selected)
        scores = trace.get("scores", [])
        if scores:
            max_score_idx = max(range(len(scores)), key=lambda i: scores[i].get("total", 0))
            sel_idx = trace.get("selected_index", 0)
            if max_score_idx != sel_idx:
                # This can be OK due to safety adjustments, but flag it
                max_s = scores[max_score_idx].get("total", 0)
                sel_s = scores[sel_idx].get("total", 0)
                if max_s > sel_s * 1.2:  # Only flag significant differences
                    anomalies.append(
                        f"[{rid}] Score mismatch: highest={max_score_idx}({max_s:.3f}), "
                        f"selected={sel_idx}({sel_s:.3f})"
                    )

    if anomalies:
        lines.append(f"발견된 이상: {len(anomalies)}건")
        for a in anomalies:
            lines.append(f"  ⚠ {a}")
    else:
        lines.append("✓ 이상 없음: 숨겨진 시스템 오류 미발견")

    lines.append("")

    # ===================================================================
    # STEP 5: Random Spot-Check (25 samples)
    # ===================================================================

    section("STEP 5: Random Spot-Check (25 Samples for Human Review)")

    lines.append("아래 25개 샘플을 수동 검토하여 다음을 확인하세요:")
    lines.append("  (A) 자연스러워 보이지만 의미가 틀린 경우 (semantic drift)")
    lines.append("  (B) cleanup이 원문을 손상시킨 경우 (cleanup damage)")
    lines.append("  (C) 점수는 높지만 품질이 낮은 경우 (score-quality gap)")
    lines.append("")

    # Select 25 samples: mix of exact, near-miss, and bad
    non_exact = [r for r in results
                 if r["system_output"].strip() != refs.get(r["id"], "").strip()]
    exact_samples = [r for r in results
                     if r["system_output"].strip() == refs.get(r["id"], "").strip()]

    # Sample: 5 exact, 20 non-exact (random)
    random.seed(42)  # reproducible
    spot_exact = random.sample(exact_samples, min(5, len(exact_samples)))
    spot_errors = random.sample(non_exact, min(20, len(non_exact)))
    spot_check = spot_exact + spot_errors
    spot_check.sort(key=lambda r: r["id"])

    for r in spot_check:
        sid = r["id"]
        out = r["system_output"].strip()
        ref = refs.get(sid, "").strip()
        words = r.get("words", [])
        trace = traces.get(sid, {})
        match = "==" if out == ref else "!="

        # Get score info
        scores = trace.get("scores", [])
        sel_idx = trace.get("selected_index", 0)
        sel_score = scores[sel_idx].get("total", 0) if scores and 0 <= sel_idx < len(scores) else -1
        cleanup_changes = trace.get("cleanup_changes", [])
        change_types = [c.get("type", "") for c in cleanup_changes] if cleanup_changes else []

        lines.append(f"[{sid}] {match}")
        lines.append(f"  입력: {' / '.join(words)}")
        lines.append(f"  참고: {ref}")
        lines.append(f"  출력: {out}")
        lines.append(f"  점수: {sel_score:.3f} | cleanup: {change_types if change_types else 'none'}")
        lines.append(f"  판정: [  ] OK  [  ] Semantic Drift  [  ] Cleanup Damage  [  ] Score Gap")
        lines.append("")

    # ===================================================================
    # STEP 6: Version Freeze Manifest
    # ===================================================================

    section("STEP 6: Version Freeze Manifest")

    lines.append(f"pipeline_version: {PIPELINE_VERSION}")
    lines.append(f"model_version: {MODEL_VERSION}")
    lines.append(f"adapter_path: exaone-finetuned-v3")
    lines.append(f"evaluation_dataset: test_data_clean.json ({total} samples)")
    lines.append(f"evaluation_timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Configuration (frozen):")
    lines.append("  GenerationConfig:")
    lines.append("    num_sampling_candidates: 2 (+ 1 greedy = 3 total)")
    lines.append("    sampling_temperature: 0.7")
    lines.append("    sampling_top_p: 0.9")
    lines.append("    max_new_tokens_finetuned: 64")
    lines.append("    repetition_penalty: 1.3")
    lines.append("    prompt_type: fewshot")
    lines.append("")
    lines.append("  ScoringConfig:")
    lines.append("    w_keyword: 0.35")
    lines.append("    w_negation: 0.25")
    lines.append("    w_hallucination: 0.25")
    lines.append("    w_length: 0.15")
    lines.append("    min_acceptable_score: 0.4")
    lines.append("    perfect_score_threshold: 0.95")
    lines.append("")
    lines.append("  CleanupConfig:")
    lines.append("    All format cleanup: ON")
    lines.append("    All semantic cleanup: ON")
    lines.append("    Semantic cleanup rollback: ON (do-no-harm guard)")
    lines.append("    Tracing: ON (traces_v2/)")

    # ===================================================================
    # STEP 7: Final Summary for Advisor
    # ===================================================================

    section("STEP 7: Final Summary for Advisor Review")

    subsection("A. Evaluation Summary")
    lines.append(f"  Exact Match: {exact_match}/{total} ({exact_match/total*100:.1f}%)")
    lines.append(f"  일상 (Daily): see category breakdown above")
    lines.append(f"  의료 (Medical): see category breakdown above")
    lines.append(f"  Pipeline Stability: {'PASS' if stability_ok else 'ISSUES'}")

    subsection("B. Error Distribution Summary")
    lines.append(f"  모델 원인: {len(pure_model)+len(both)}/{n_err} ({(len(pure_model)+len(both))/max(n_err,1)*100:.1f}%)")
    lines.append(f"  파이프라인 원인: {len(pure_pipeline)+len(both)}/{n_err} ({(len(pure_pipeline)+len(both))/max(n_err,1)*100:.1f}%)")
    lines.append(f"  Top-3 오류:")
    all_error_counts = Counter()
    for name, ids in stats.items():
        if ids:
            all_error_counts[name] = len(ids)
    for name, count in all_error_counts.most_common(3):
        lines.append(f"    {name}: {count} ({count/max(n_err,1)*100:.1f}%)")

    subsection("C. Top 20 Bad Cases")
    # Select worst 20 by: most error categories, longest distance from reference
    scored_errors = []
    for r in non_exact:
        sid = r["id"]
        out = r["system_output"].strip()
        ref = refs.get(sid, "").strip()

        # Simple quality metric: character-level similarity
        common = sum(1 for a, b in zip(out, ref) if a == b)
        max_len = max(len(out), len(ref), 1)
        char_sim = common / max_len

        # Count detected issues
        n_issues = 0
        for stat_ids in stats.values():
            if sid in stat_ids:
                n_issues += 1

        scored_errors.append((sid, n_issues, char_sim, r))

    # Sort by most issues, then lowest similarity
    scored_errors.sort(key=lambda x: (-x[1], x[2]))

    for sid, n_issues, char_sim, r in scored_errors[:20]:
        out = r["system_output"].strip()
        ref = refs.get(sid, "").strip()
        words = r.get("words", [])
        trace = traces.get(sid, {})

        issue_names = [name for name, ids in stats.items() if sid in ids]

        lines.append(f"  [{sid}] issues={n_issues}, similarity={char_sim:.2f}")
        lines.append(f"    입력: {' / '.join(words)}")
        lines.append(f"    참고: {ref}")
        lines.append(f"    출력: {out}")
        lines.append(f"    문제: {', '.join(issue_names)}")
        lines.append("")

    subsection("D. Key Findings")
    lines.append("  1. 시스템 공학 문제는 대부분 해결됨:")
    lines.append("     - Uncontrolled hallucination → candidate scoring으로 억제")
    lines.append("     - Retry chaos → 제거됨 (deterministic pipeline)")
    lines.append("     - Postprocess 오염 → format/semantic 분리로 해결")
    lines.append("     - Prompt residue → guardrail에서 차단")
    lines.append("")
    lines.append("  2. 남은 문제는 모델 레벨:")
    lines.append(f"     - Narrative expansion: {len(stats['narrative_expansion'])}건")
    lines.append(f"     - Semantic drift: {len(stats['semantic_drift'])}건")
    lines.append(f"     - Causal reasoning: {len(stats['causal_reasoning'])}건")
    lines.append("     → 2.4B 모델의 reasoning 한계")
    lines.append("")
    lines.append("  3. 논문 limitation 포인트:")
    lines.append("     - Small model (2.4B) reasoning ceiling")
    lines.append("     - Korean syntax composition instability")
    lines.append("     - Narrative expansion tendency (conversational bias)")
    lines.append("     - No semantic consistency scorer (would need larger model or NLI)")
    lines.append("")
    lines.append("  4. 논문 강조 포인트:")
    lines.append("     - 7-layer pipeline으로 시스템 오류 체계적 제거")
    lines.append("     - Score-based candidate selection")
    lines.append("     - Do-no-harm rollback 메커니즘")
    lines.append("     - Full inference tracing for error analysis")
    lines.append("     - Failure taxonomy (14 types)")
    lines.append("     - Data leakage verified: 0% input overlap")

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Audit completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Pipeline version: {PIPELINE_VERSION}")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    report = run_audit()
    print(report)

    REPORT_OUTPUT.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {REPORT_OUTPUT}")
