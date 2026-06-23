"""Final merge: combine all data sources into training-ready files.

Run AFTER generate_data_v3_supplement.py completes.
Output: train_data_all_final.json + train_data_all_final_chat.jsonl
"""
import json
import sys
from pathlib import Path
from refilter_all_data import validate_entry

BASE_DIR = Path(__file__).resolve().parent

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    sources = [
        ("Merged v3 (6612)", "train_data_final_merged.json"),
        ("Supplement", "train_data_supplement.json"),
    ]

    refs = set()
    merged = []

    for label, fname in sources:
        p = BASE_DIR / fname
        if not p.exists():
            print(f"  SKIP {label}: {fname} not found")
            continue
        data = json.load(open(p, "r", encoding="utf-8"))
        added = 0
        for e in data:
            if not isinstance(e, dict) or "reference" not in e:
                continue
            if e["reference"] not in refs:
                ok, _ = validate_entry(e)
                if ok:
                    merged.append(e)
                    refs.add(e["reference"])
                    added += 1
        print(f"  {label}: {len(data)} → {added} added (after dedup+validate)")

    print(f"\nTOTAL: {len(merged)} entries")

    # Word count distribution
    from collections import Counter
    wc = Counter(len(e["words"]) for e in merged)
    print("\nWord count distribution:")
    for k in sorted(wc.keys()):
        print(f"  {k} words: {wc[k]} entries")

    # Save JSON
    out_json = BASE_DIR / "train_data_all_final.json"
    out_json.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save chat JSONL
    out_jsonl = BASE_DIR / "train_data_all_final_chat.jsonl"
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for e in merged:
            chat = {
                "messages": [
                    {"role": "system", "content": "단어 시퀀스를 자연스러운 한국어 문장 하나로 변환하라. 반드시 해라체(-다, -ㄴ다, -는다)만 사용하고, 입력에 없는 단어를 추가하지 마라."},
                    {"role": "user", "content": "입력 단어: " + " / ".join(e["words"])},
                    {"role": "assistant", "content": e["reference"]}
                ]
            }
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_jsonl}")
    print("Ready for training!")

if __name__ == "__main__":
    main()
