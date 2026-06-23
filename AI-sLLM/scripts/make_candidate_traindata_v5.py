# -*- coding: utf-8 -*-
"""Candidate-format training data built on the CLEANED v5 data (task #1, v6).

Same scheme as make_candidate_traindata.py (50% candidate / 50% clean, each
candidate slot = 1 true word + 2 distractors, shuffled), BUT the source is the
hallucination-cleaned conservative set (train_data_all_final_chat_v5_conservative.jsonl)
instead of the dirty train_data_all_final.json. So the candidate model (v6)
also inherits the v5 cleanup. Format is byte-identical to v4-cand so the
recovery number stays comparable to the 79.8% baseline.

Usage: python scripts/make_candidate_traindata_v5.py
Out:   train_data_candidate_v5_chat.jsonl
"""
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from prompt_templates import build_candidate_string  # noqa: E402

SRC = os.path.join(HERE, "train_data_all_final_chat_v5_conservative.jsonl")
OUT = os.path.join(HERE, "train_data_candidate_v5_chat.jsonl")
SEED, N_CAND, P_CAND = 42, 3, 0.5
random.seed(SEED)

# Must match FinetunedGenerator's system message exactly (train == inference).
SYSTEM = ("단어 시퀀스를 자연스러운 한국어 문장 하나로 변환하라. "
          "반드시 합쇼체(-습니다, -ㅂ니다)만 사용하고, 입력에 없는 단어를 추가하지 마라. "
          "반드시 한국어만 출력하고 영어 단어를 사용하지 마라.")


def parse(obj):
    words, ref = [], ""
    for m in obj["messages"]:
        if m["role"] == "user":
            c = m["content"].split("입력 단어:")[-1]
            words = [x.strip() for x in c.split("/") if x.strip()]
        elif m["role"] == "assistant":
            ref = m["content"].strip()
    return words, ref


items = []
for ln in open(SRC, encoding="utf-8"):
    ln = ln.strip()
    if ln:
        w, r = parse(json.loads(ln))
        if w and r:
            items.append((w, r))


def is_content(w):
    return len(w) >= 2 and not re.fullmatch(r"[0-9]+", w)


vocab = sorted({w for ws, _ in items for w in ws if is_content(w)})


def distractors(true, k=2):
    pool = [w for w in vocab if w != true and (set(w) & set(true) or abs(len(w) - len(true)) <= 1)]
    if len(pool) < k:
        pool = [w for w in vocab if w != true]
    return random.sample(pool, k)


n_cand = n_clean = 0
with open(OUT, "w", encoding="utf-8") as f:
    for words, ref in items:
        if random.random() < P_CAND:
            cands = []
            for w in words:
                cs = [w] + distractors(w, N_CAND - 1)
                random.shuffle(cs)
                cands.append(cs)
            user = "입력 단어: " + build_candidate_string(cands)
            n_cand += 1
        else:
            user = "입력 단어: " + " / ".join(words)
            n_clean += 1
        f.write(json.dumps({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ref},
        ]}, ensure_ascii=False) + "\n")

print(f"wrote {os.path.basename(OUT)}")
print(f"  source={os.path.basename(SRC)}  items={len(items)}")
print(f"  candidate={n_cand}  clean={n_clean}  total={n_cand + n_clean}")
