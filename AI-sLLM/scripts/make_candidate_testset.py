# -*- coding: utf-8 -*-
"""Build a candidate-set test for task #1 (SLR-noise robustness experiment).

For each test item (words + reference), produce per-position candidate sets:
  1 true word + 2 distractors, shuffled. Also produce a 'noisy top-1' version
  (each position has prob P_ERR of being replaced by a distractor) for the
  baseline condition.

No real SLR data is available -> distractors are synthesized from the dataset
vocabulary (content words), mildly biased toward orthographic/length similarity
so they are non-trivial. This is an APPROXIMATION of SLR confusion (which is
really visual, not lexical); real SLR top-k would be stronger. Seed is fixed
for reproducibility.

Usage:  python scripts/make_candidate_testset.py
Output: candidate_test.json
"""
import json, io, sys, os, random, re

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED = 42
N_CAND = 3       # candidates per position (1 true + 2 distractors)
P_ERR = 0.3      # baseline: probability a position's top-1 is wrong
random.seed(SEED)


def load(p):
    return json.load(open(os.path.join(HERE, p), encoding="utf-8"))


train = load("train_data_all_final.json")
test = load("test_data_clean.json")


def is_content(w):
    return len(w) >= 2 and not re.fullmatch(r"[0-9]+", w)


vocab = sorted({w for it in (train + test) for w in it["words"] if is_content(w)})


def distractors(true, k=2):
    """Pick k distractors: prefer words sharing a char with `true` or similar length."""
    pool_sim = [w for w in vocab
                if w != true and (set(w) & set(true) or abs(len(w) - len(true)) <= 1)]
    pool = pool_sim if len(pool_sim) >= k else [w for w in vocab if w != true]
    return random.sample(pool, k)


out = []
for it in test:
    words = it["words"]
    cands, noisy = [], []
    for w in words:
        ds = distractors(w, N_CAND - 1)
        cset = [w] + ds
        random.shuffle(cset)
        cands.append(cset)
        noisy.append(random.choice(ds) if random.random() < P_ERR else w)
    out.append({
        "request_id": it.get("request_id", ""),
        "domain": it.get("domain", ""),
        "words": words,            # ground-truth words
        "reference": it["reference"],
        "candidates": cands,       # per-position [shuffled: true + distractors]
        "noisy_top1": noisy,       # baseline: single (possibly wrong) word per position
    })

dst = os.path.join(HERE, "candidate_test.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

n_pos = sum(len(it["words"]) for it in out)
n_corrupt = sum(1 for it in out for t, n in zip(it["words"], it["noisy_top1"]) if t != n)
print(f"vocab(content)={len(vocab)}  items={len(out)}  positions={n_pos}")
print(f"baseline corrupted positions: {n_corrupt}/{n_pos} ({n_corrupt/n_pos:.1%})")
print("\nsample item:")
print(json.dumps(out[0], ensure_ascii=False, indent=2))
