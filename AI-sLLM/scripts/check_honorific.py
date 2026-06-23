# -*- coding: utf-8 -*-
"""Report the honorific (합쇼체) rate of predictions in an eval-results JSON.

    python -X utf8 scripts/check_honorific.py --results evaluation_results_v5_cons.json
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
args = ap.parse_args()

data = json.load(open(args.results, encoding="utf-8"))
rows = data["rows"] if isinstance(data, dict) and "rows" in data else data


def is_honorific(text: str) -> bool:
    t = (text or "").rstrip(" .?!。")
    return t.endswith("니다") or t.endswith("니까")


pred_key = "prediction"
hon = 0
nonhon = []
for r in rows:
    p = r.get("prediction") or r.get("system_output") or ""
    if is_honorific(p):
        hon += 1
    else:
        nonhon.append((r.get("id", "?"), p))

n = len(rows)
print(f"honorific(합쇼체): {hon}/{n} = {hon / n * 100:.1f}%")
for rid, p in nonhon:
    print(f"  non-hon [{rid}]: {p}")
