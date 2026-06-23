# -*- coding: utf-8 -*-
"""Generate candidate-format training data for task #1 fine-tuning.

For each training item, with prob P_CAND emit the CANDIDATE format
(per-position true word + distractors, shuffled), else the CLEAN format
(single true words). Target is always the correct honorific reference.
One dataset teaches both clean conversion and candidate disambiguation, so the
fine-tuned model handles real single-word input AND noisy SLR candidate sets.

Distractors are synthesized from the dataset vocabulary (same method as the
candidate test set) — an approximation of SLR confusion. Seed is fixed.

Usage:  python scripts/make_candidate_traindata.py
Output: train_data_candidate_chat.jsonl
"""
import json, io, sys, os, random, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from prompt_templates import build_candidate_string

SEED = 42
N_CAND = 3      # candidates per position (1 true + 2 distractors)
P_CAND = 0.5    # fraction of items rendered in candidate format
random.seed(SEED)

# Must match FinetunedGenerator's system message exactly (train == inference).
SYSTEM = ("단어 시퀀스를 자연스러운 한국어 문장 하나로 변환하라. "
          "반드시 합쇼체(-습니다, -ㅂ니다)만 사용하고, 입력에 없는 단어를 추가하지 마라. "
          "반드시 한국어만 출력하고 영어 단어를 사용하지 마라.")

train = json.load(open(os.path.join(HERE, "train_data_all_final.json"), encoding="utf-8"))


def is_content(w):
    return len(w) >= 2 and not re.fullmatch(r"[0-9]+", w)


vocab = sorted({w for it in train for w in it["words"] if is_content(w)})


def distractors(true, k=2):
    pool = [w for w in vocab if w != true and (set(w) & set(true) or abs(len(w) - len(true)) <= 1)]
    if len(pool) < k:
        pool = [w for w in vocab if w != true]
    return random.sample(pool, k)


n_cand = n_clean = 0
samples = []
out_path = os.path.join(HERE, "train_data_candidate_chat.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for it in train:
        words, ref = it["words"], it["reference"]
        if random.random() < P_CAND:
            cands = []
            for w in words:
                cs = [w] + distractors(w, N_CAND - 1)
                random.shuffle(cs)
                cands.append(cs)
            user = "입력 단어: " + build_candidate_string(cands)
            n_cand += 1
            tag = "CAND"
        else:
            user = "입력 단어: " + " / ".join(words)
            n_clean += 1
            tag = "CLEAN"
        rec = {"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ref},
        ]}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if len([s for s in samples if s[0] == tag]) < 1:
            samples.append((tag, user, ref))

print(f"wrote {out_path}")
print(f"  candidate={n_cand}  clean={n_clean}  total={n_cand + n_clean}")
print("\n--- samples ---")
for tag, user, ref in samples:
    print(f"[{tag}] user: {user}")
    print(f"       asst: {ref}")
