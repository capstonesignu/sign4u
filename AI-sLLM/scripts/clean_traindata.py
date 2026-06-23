# -*- coding: utf-8 -*-
"""Clean hallucination-teaching examples from the chat training data.

The fine-tune system prompt says "입력에 없는 단어를 추가하지 마라" (do NOT add words
not in the input) — yet ~11% of the target sentences add content anyway. That
trains the model to disobey its own instruction. This script flags those targets
(canonical metric, same definition as reaudit.py) and writes cleaned datasets so
we can retrain a faithful model.

Buckets (by what the TARGET adds beyond particles / grammar / light verbs):
  clean            : adds nothing                                    -> keep
  actor            : invents a person (의사/환자/간호사/...)            -> drop
  extra            : invents a content noun/adjective (수정/심하/주의)  -> drop
  forced_predicate : adds only a verb to turn a noun-bag input into a
                     sentence (알리다/따르다 ...)                       -> kept by default

Outputs (non-destructive, next to the input file):
  *_v5_conservative.jsonl : drop actor+extra, keep clean+forced_predicate
  *_v5_aggressive.jsonl   : keep clean only

    python scripts/clean_traindata.py --data train_data_all_final_chat.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reaudit import input_lemma_set, _morphemes, CONTENT_POS, LIGHT  # noqa: E402

ACTORS = {
    "의사", "환자", "간호사", "사람", "보호자", "가족", "직원", "상담원",
    "약사", "아이", "선생", "엄마", "아빠", "어머니", "아버지", "친구", "동료",
}


def parse_line(obj: dict) -> tuple[list[str], str]:
    """Extract (input words, target sentence) from a chat-format line."""
    words: list[str] = []
    target = ""
    for m in obj.get("messages", []):
        if m.get("role") == "user":
            c = m.get("content", "")
            if "입력 단어:" in c:
                c = c.split("입력 단어:")[-1]
            words = [w.strip() for w in c.split("/") if w.strip()]
        elif m.get("role") == "assistant":
            target = m.get("content", "").strip()
    return words, target


def added_tagged(words: list[str], pred: str) -> list[tuple[str, str]]:
    inp = input_lemma_set(words)
    return [
        (t.form, t.tag)
        for t in _morphemes(pred)
        if t.tag in CONTENT_POS
        and len(t.form) >= 2
        and t.form not in LIGHT
        and t.form not in inp
        # skip compound-split false positives: 의사 inside 의사소통, 약 inside 예약...
        and not any(t.form in iw for iw in words)
    ]


def bucket_of(words: list[str], target: str) -> tuple[str, list[str]]:
    add = added_tagged(words, target)
    if not add:
        return "clean", []
    forms = [f for f, _ in add]
    if any(f in ACTORS for f in forms):
        return "actor", forms
    if all(tag == "VV" for _, tag in add):
        return "forced_predicate", forms
    return "extra", forms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_data_all_final_chat.jsonl")
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    path = Path(args.data)
    raw_lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    buckets: Counter = Counter()
    tagged: list[tuple[str, str]] = []          # (bucket, raw_line)
    samples: dict[str, list] = {"actor": [], "extra": [], "forced_predicate": []}

    for raw in raw_lines:
        obj = json.loads(raw)
        w, t = parse_line(obj)
        b, forms = bucket_of(w, t)
        buckets[b] += 1
        tagged.append((b, raw))
        if b in samples and len(samples[b]) < args.show:
            samples[b].append((w, t, forms))

    n = len(raw_lines)
    print("=" * 60)
    print(f"CLEAN TRAIN DATA  ·  {path.name}  ·  {n} examples")
    print("=" * 60)
    for b in ("clean", "forced_predicate", "extra", "actor"):
        print(f"  {b:18s}: {buckets[b]:5d}  ({buckets[b] / n * 100:.1f}%)")

    cons = [raw for b, raw in tagged if b in ("clean", "forced_predicate")]
    aggr = [raw for b, raw in tagged if b == "clean"]
    cons_path = path.with_name(path.stem + "_v5_conservative.jsonl")
    aggr_path = path.with_name(path.stem + "_v5_aggressive.jsonl")
    cons_path.write_text("\n".join(cons) + "\n", encoding="utf-8")
    aggr_path.write_text("\n".join(aggr) + "\n", encoding="utf-8")

    print()
    print(f"  conservative (drop actor+extra) -> {len(cons):5d} kept  -> {cons_path.name}")
    print(f"  aggressive   (drop all dirty)    -> {len(aggr):5d} kept  -> {aggr_path.name}")

    for b in ("actor", "extra", "forced_predicate"):
        print(f"\n--- {b} samples ---")
        for w, t, f in samples[b]:
            print("  in:", " / ".join(w), " tgt:", t, " +:", ",".join(f))


if __name__ == "__main__":
    main()
