# -*- coding: utf-8 -*-
"""Task #1 robustness experiment (mechanism B).

3 input conditions: clean / noisy_top1 / candidate. Two backends:
  --backend exaone    : BASE instruction-following model + full instruction prompts
  --backend finetuned : candidate-format fine-tuned model + compact format (train==infer)

The 3 conditions share one model, so the comparison (does giving candidates beat
noisy top-1?) is the signal. Needs GPU. Examples:

  python scripts/eval_candidates.py --backend finetuned --adapter exaone-finetuned-v4-cand
  python scripts/eval_candidates.py --backend exaone            # base 2.4B
  python scripts/eval_candidates.py --limit 20                  # quick subset
"""
import json, io, sys, os, argparse, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.chdir(HERE)

from prompt_templates import build_fewshot_prompt, build_candidate_prompt, build_candidate_string
from format_cleanup import FormatCleaner
from config import CleanupConfig
from evaluate import sentence_bleu, chrf_plusplus, keyword_recall


def is_hon(t):
    t = t.rstrip()
    return t.endswith(("습니다.", "ㅂ니다.", "습니까?", "ㅂ니까?", "니다.", "니까?"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="exaone", choices=["exaone", "finetuned"])
    ap.add_argument("--model-id", default="", help="base model id override")
    ap.add_argument("--adapter", default="", help="LoRA adapter (finetuned backend)")
    ap.add_argument("--data", default="candidate_test.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="candidate_eval_results.json")
    args = ap.parse_args()

    data = json.load(open(args.data, encoding="utf-8"))
    if args.limit:
        data = data[:args.limit]

    if args.backend == "finetuned":
        from sllm_module import FinetunedGenerator
        gen = FinetunedGenerator(adapter_path=args.adapter, model_id=args.model_id)

        def prompts(it):  # compact format — matches candidate-format fine-tuning
            return {
                "clean": " / ".join(it["words"]),
                "noisy_top1": " / ".join(it["noisy_top1"]),
                "candidate": build_candidate_string(it["candidates"]),
            }
    else:
        from sllm_module import ExaoneGenerator
        gen = ExaoneGenerator(model_id=args.model_id)

        def prompts(it):  # full instruction prompts for a base instruction-follower
            dom = it.get("domain", "daily")
            return {
                "clean": build_fewshot_prompt(it["words"], dom),
                "noisy_top1": build_fewshot_prompt(it["noisy_top1"], dom),
                "candidate": build_candidate_prompt(it["candidates"], dom),
            }

    fc = FormatCleaner(CleanupConfig())

    def run(prompt, words):
        return fc.clean(gen.generate(words, prompt=prompt))

    conds = ["clean", "noisy_top1", "candidate"]
    agg = {c: {"bleu": 0.0, "chrf": 0.0, "recall": 0.0, "hon": 0.0} for c in conds}
    rows = []
    t0 = time.time()
    for i, it in enumerate(data):
        true, ref = it["words"], it["reference"]
        pr = prompts(it)
        outs = {c: run(pr[c], true) for c in conds}
        row = {"ref": ref, "true": true}
        for c in conds:
            p = outs[c]
            m = {"pred": p, "bleu": sentence_bleu(p, ref), "chrf": chrf_plusplus(p, ref),
                 "recall": keyword_recall(true, p), "hon": 1.0 if is_hon(p) else 0.0}
            row[c] = m
            for k in agg[c]:
                agg[c][k] += m[k]
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(data)}", flush=True)

    n = len(rows)
    for c in conds:
        for k in agg[c]:
            agg[c][k] /= n
    dt = time.time() - t0
    print(f"\nbackend={args.backend} adapter={args.adapter or '-'} model={args.model_id or 'default'}")
    print(f"items={n}  time={dt:.1f}s ({dt / n:.2f}s/item x3)")
    print(f"{'condition':12}{'BLEU':>8}{'chrF++':>8}{'true-recall':>13}{'honorific':>11}")
    for c in conds:
        a = agg[c]
        print(f"{c:12}{a['bleu']:8.4f}{a['chrf']:8.4f}{a['recall']:12.1%}{a['hon']:11.1%}")
    print("\ncandidate should beat noisy_top1 if the model learned to pick from candidates.")
    json.dump({"agg": agg, "rows": rows}, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("saved", args.output)


if __name__ == "__main__":
    main()
