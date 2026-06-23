# -*- coding: utf-8 -*-
"""Check honorific (합쇼체) rate of evaluation predictions + print BLEU/chrF.

Reads evaluation_results_v4.json produced by:
  python evaluate.py --backend finetuned --data test_data_clean.json --output evaluation_results_v4.json
"""
import json, io, sys, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "evaluation_results_v4.json")
res = json.load(open(RESULT, encoding="utf-8"))
rows = res["rows"]
preds = [r["prediction"] for r in rows]

print(f"BLEU={res['avg_sentence_bleu']:.4f}  chrF++={res['avg_chrf_plusplus']:.4f}  "
      f"kw_recall={res['avg_keyword_recall']:.2%}  exact={res['exact_match']:.2%}  "
      f"halluc={res['avg_hallucination_rate']:.2%}")

def is_hon(t):
    t = t.rstrip()
    return t.endswith(("습니다.", "ㅂ니다.", "습니까?", "ㅂ니까?", "니다.", "니까?"))

hon = sum(1 for p in preds if is_hon(p))
print(f"predictions={len(preds)}  honorific={hon}  rate={hon/max(1,len(preds)):.1%}")
print("--- non-honorific predictions (if any) ---")
nh = [(r["words"], r["prediction"]) for r in rows if not is_hon(r["prediction"])]
for w, p in nh[:15]:
    print("   ", w, "->", p)
print("--- sample (first 6) ---")
for p in preds[:6]:
    print("   -", p)
