"""Analyze hallucination cases in detail."""
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

data = json.loads(open("evaluation_v3_morpho.json", "r", encoding="utf-8").read())

print("=" * 70)
print("HALLUCINATION CASES (26/100)")
print("=" * 70)

count = 0
for r in data["rows"]:
    if "hallucination" in r["errors"]:
        count += 1
        w = " / ".join(r["words"])
        p = r["prediction"]
        ref = r["reference"]
        print(f"\n[{count}] Input:  {w}")
        print(f"    Output: {p}")
        print(f"    Ref:    {ref}")

print(f"\nTotal: {count}")
