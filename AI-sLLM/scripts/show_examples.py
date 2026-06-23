"""Show diverse example outputs for report."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

data = json.loads(open("evaluation_v3_morpho.json", "r", encoding="utf-8").read())
rows = data["rows"]

indices = [0, 5, 10, 15, 27, 35, 50, 75]
for i in indices:
    if i < len(rows):
        r = rows[i]
        w = r["words"]
        p = r["prediction"]
        ref = r["reference"]
        kw = r["keyword_recall"]
        err = r["errors"]
        print(f"[{i+1}] Input:  {w}")
        print(f"    Output: {p}")
        print(f"    Ref:    {ref}")
        print(f"    KW: {kw:.0%}  Errors: {err}")
        print()
