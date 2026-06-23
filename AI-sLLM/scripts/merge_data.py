"""Merge all validated training data into final training files."""
import json
import sys

def get_stem(word):
    if word.endswith("하다"): return word[:-2]
    if word.endswith("다") and len(word) >= 2: return word[:-1]
    return word

def word_in_ref(word, ref):
    if word in ref: return True
    stem = get_stem(word)
    if len(stem) >= 2 and stem in ref: return True
    if len(word) == 1 and word in ref: return True
    return False

def validate(entry):
    words = entry.get("words", [])
    ref = entry.get("reference", "")
    if not words or not ref or len(words) < 2: return False
    ref_s = ref.rstrip(".!?")
    if any(ref_s.endswith(e) for e in ["요","습니다","세요","해요"]): return False
    for w in words:
        if not word_in_ref(w, ref): return False
    return True

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Load all sources
    r1 = json.load(open("train_data_v3.json", "r", encoding="utf-8"))
    print(f"Round 1: {len(r1)} entries")

    r2 = json.load(open("train_data_v3_r2.json", "r", encoding="utf-8"))
    print(f"Round 2: {len(r2)} entries")

    old = json.load(open("train_data_5k.json", "r", encoding="utf-8"))
    old_pass = [e for e in old if validate(e)]
    print(f"Old 5k filtered: {len(old_pass)}/{len(old)} entries")

    # Merge with dedup (new data priority)
    refs = set()
    merged = []
    for source, data in [("R1", r1), ("R2", r2), ("Old", old_pass)]:
        count = 0
        for e in data:
            if e["reference"] not in refs:
                merged.append(e)
                refs.add(e["reference"])
                count += 1
        print(f"  Added from {source}: {count} (after dedup)")

    print(f"\nTotal merged: {len(merged)} entries")

    # Save JSON
    with open("train_data_merged_v3.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # Save chat JSONL for training
    with open("train_data_merged_v3_chat.jsonl", "w", encoding="utf-8") as f:
        for e in merged:
            chat = {
                "messages": [
                    {"role": "system", "content": "단어 시퀀스를 자연스러운 한국어 문장 하나로 변환하라. 반드시 해라체(-다, -ㄴ다, -는다)만 사용하고, 입력에 없는 단어를 추가하지 마라."},
                    {"role": "user", "content": "입력 단어: " + " / ".join(e["words"])},
                    {"role": "assistant", "content": e["reference"]}
                ]
            }
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")

    print("Saved: train_data_merged_v3.json")
    print("Saved: train_data_merged_v3_chat.jsonl")
    print("Ready for training!")

if __name__ == "__main__":
    main()
