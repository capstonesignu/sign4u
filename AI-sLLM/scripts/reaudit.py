# -*- coding: utf-8 -*-
"""Re-audit a saved evaluation result with corrected, in-code metrics.

NO GPU / NO API needed. Runs on any saved eval-results JSON whose rows carry
``words`` / ``prediction`` (or ``system_output``) / ``reference``. It exposes the
"true quality" that single-reference exact-match and BLEU hide:

  * bucket every non-exact output into a *likely valid paraphrase* (heuristic)
    vs the real error types (spacing / english-leak / narrative-expansion /
    causal) by reusing the existing detectors in ``error_analysis.py``;
  * compute ONE canonical hallucination number (kiwi POS based: a content
    morpheme whose lemma is not derivable from the input glosses) — no
    hardcoded adverb blocklist, no substring false positives. This replaces
    the non-reproducible "54%" headline with something defined in code.

This is the lean evaluation "ruler" step 0: it tells you, on data you already
have, how much of the apparent error rate is just paraphrase punishment.

Usage (run where the repo + kiwipiepy live, e.g. D:\\SLLM\\AI-sLLM):
    python scripts/reaudit.py --results eval_metrics_finetuned.json
    python scripts/reaudit.py --results evaluation_v3_morpho.json --show 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# NOTE: error_analysis (imported below) reconfigures stdout to UTF-8 on import.
# We must NOT wrap stdout a second time, or the shared buffer gets closed.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the already-validated detectors (same logic that produced the existing
# error_analysis_report.txt) instead of re-implementing them.
from error_analysis import (  # noqa: E402
    detect_english_leak,
    detect_narrative_expansion,
    detect_causal_error,
)

# ---------------------------------------------------------------------------
# Canonical hallucination — the ONE definition we will use everywhere.
# ---------------------------------------------------------------------------

# kiwi POS tags counted as "content": general/proper noun, verb, adjective,
# general adverb, root. Particles, endings, connectives are NOT here, so the
# grammatical glue that naturalisation adds (는/을/어서/니까 ...) is ignored.
CONTENT_POS = {"NNG", "NNP", "VV", "VA", "MAG", "XR"}

# Light/structural verbs that turn an input noun into a predicate. Adding these
# is grammar, not hallucination (e.g. input "준비" -> "준비를 한다").
# Light/structural verbs + grammatical connectives whose lemma kiwi surfaces as
# a "content" POS but which add no real content (위해="in order to", etc.).
LIGHT = {"하다", "되다", "있다", "않다", "같다", "위하", "위하다", "드리다", "주다", "지다"}

_kiwi = None


def _kiwi_inst():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def _morphemes(text: str) -> list:
    try:
        return _kiwi_inst().analyze(text)[0][0]
    except Exception:
        return []


def input_lemma_set(words: list[str]) -> set[str]:
    """All forms the input glosses can legitimately surface as."""
    inp: set[str] = set(words)
    # "X하다 / X되다 / X받다" glosses faithfully surface as the noun root "X"
    # (예약하다 -> 예약, 확인하다 -> 확인): strip the verbaliser so we don't flag
    # the noun form as hallucinated.
    verbalisers = ("하다", "되다", "시키다", "당하다", "받다", "하", "되")
    for w in words:
        for tok in _morphemes(w):
            inp.add(tok.form)
        if w.endswith("다") and len(w) > 1:
            inp.add(w[:-1])
        for suf in verbalisers:
            if w.endswith(suf) and len(w) > len(suf):
                inp.add(w[: -len(suf)])
    return inp


def canonical_hallucinations(words: list[str], pred: str) -> list[str]:
    """Content morphemes in ``pred`` not derivable from the input glosses."""
    inp = input_lemma_set(words)
    out: list[str] = []
    for tok in _morphemes(pred):
        if (
            tok.tag in CONTENT_POS
            and len(tok.form) >= 2
            and tok.form not in LIGHT
            and tok.form not in inp
        ):
            out.append(tok.form)
    return out


def has_spacing_error(pred: str, min_missing: int = 2) -> bool:
    """True if kiwi's auto-spacer would insert >= min_missing spaces.

    Replaces the old "6+ Korean chars in a row" heuristic, which false-flagged
    legitimately long honorific words (접수했습니다 = 6 chars, no space needed).
    min_missing=2 ignores single optional-space differences (보조용언 etc.) and
    only catches genuine run-ons (발이삐서걷때조심했습니다).
    """
    p = pred.strip()
    if not p:
        return False
    try:
        fixed = _kiwi_inst().space(p)
    except Exception:
        return False
    if fixed.replace(" ", "") != p.replace(" ", ""):
        return False  # spacer altered non-space content; don't trust it
    return (fixed.count(" ") - p.count(" ")) >= min_missing


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

ERROR_TAGS = ["spacing", "english_leak", "narrative_expansion", "causal", "hallucination"]


def classify(row: dict) -> tuple[list[str], list[str]]:
    """Return (tags, hallucinated_tokens). tags is one of:
    ['exact'] | ['likely_paraphrase'] | ['low_recall_other'] | [error tags...]."""
    words = row.get("words", [])
    pred = (row.get("prediction") or row.get("system_output") or "").strip()
    ref = (row.get("reference") or "").strip()

    if pred and pred == ref:
        return ["exact"], []

    tags: list[str] = []
    if has_spacing_error(pred):
        tags.append("spacing")
    if detect_english_leak(pred):
        tags.append("english_leak")
    if detect_narrative_expansion(pred, words):
        tags.append("narrative_expansion")
    if detect_causal_error(pred, ref, words):
        tags.append("causal")
    halluc = canonical_hallucinations(words, pred)
    if halluc:
        tags.append("hallucination")

    if tags:
        return tags, halluc

    # Not exact, but no error pattern fired -> probably a valid paraphrase.
    kw = row.get("keyword_recall")
    if kw is None:
        kw = 1.0
    return (["likely_paraphrase"] if kw >= 0.8 else ["low_recall_other"]), halluc


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Re-audit a saved eval result (no GPU).")
    ap.add_argument("--results", default="evaluation/eval_metrics_finetuned.json")
    ap.add_argument("--show", type=int, default=25,
                    help="how many flagged real-error rows to print")
    args = ap.parse_args()

    path = Path(args.results)
    if not path.exists():
        print(f"[reaudit] not found: {path.resolve()}")
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    n = len(rows)

    tag_counts: Counter = Counter()
    halluc_sents = 0
    halluc_tokens: Counter = Counter()
    real_error_rows = []

    for row in rows:
        tags, halluc = classify(row)
        for t in tags:
            tag_counts[t] += 1
        if halluc:
            halluc_sents += 1
            halluc_tokens.update(halluc)
        if any(t in ERROR_TAGS for t in tags):
            real_error_rows.append((row, tags, halluc))

    exact = tag_counts.get("exact", 0)
    para = tag_counts.get("likely_paraphrase", 0)
    low = tag_counts.get("low_recall_other", 0)
    acceptable_est = exact + para

    print("=" * 64)
    print(f"RE-AUDIT  ·  {path.name}  ·  {n} samples")
    print("=" * 64)
    print(f"  exact match            : {exact:3d}/{n}  ({exact/n*100:.1f}%)")
    print(f"  likely valid paraphrase: {para:3d}/{n}  ({para/n*100:.1f}%)   [heuristic — judge to confirm]")
    print(f"  --> est. ACCEPTABLE    : {acceptable_est:3d}/{n}  ({acceptable_est/n*100:.1f}%)")
    print(f"  low-recall / other     : {low:3d}/{n}  ({low/n*100:.1f}%)")
    print()
    print("  real-error buckets (a sentence can hit several):")
    for t in ERROR_TAGS:
        c = tag_counts.get(t, 0)
        print(f"    {t:20s}: {c:3d}  ({c/n*100:.1f}%)")
    print()
    print("  CANONICAL hallucination (kiwi POS, code-defined, reproducible):")
    print(f"    sentences with >=1   : {halluc_sents:3d}/{n}  ({halluc_sents/n*100:.1f}%)")
    if halluc_tokens:
        top = ", ".join(f"{w}×{c}" for w, c in halluc_tokens.most_common(12))
        print(f"    top added tokens     : {top}")
    print(f"    (compare: progress-report headline 54-57%  vs  saved error_analysis 6.25%)")
    print()

    print("-" * 64)
    print(f"REAL-ERROR ROWS (first {args.show})")
    print("-" * 64)
    for row, tags, halluc in real_error_rows[:args.show]:
        sid = row.get("id", "?")
        et = "+".join(t for t in tags if t in ERROR_TAGS)
        print(f"  [{sid}] {et}{'  halluc=' + ','.join(halluc) if halluc else ''}")
        print(f"    in : {' / '.join(row.get('words', []))}")
        print(f"    ref: {row.get('reference','')}")
        print(f"    out: {row.get('prediction') or row.get('system_output','')}")
    print()
    print(f"[reaudit] {len(real_error_rows)} rows have >=1 real error; "
          f"{n - len(real_error_rows)} are exact/paraphrase/low-recall.")


if __name__ == "__main__":
    main()
