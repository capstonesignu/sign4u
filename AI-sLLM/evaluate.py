"""Advanced evaluation for AI-sLLM: medical-info-preservation metrics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import sacrebleu

from sllm_module import SLLMWordsToText
from medical_vocab import MEDICAL_CLINICAL, MEDICAL_BROAD

# ---------------------------------------------------------------------------
# Basic tokenization
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return text.replace(".", " .").replace("?", " ?").replace("!", " !").split()



# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def exact_match(pred: str, ref: str) -> bool:
    return pred == ref


def token_overlap(pred: str, ref: str) -> float:
    p = Counter(tokenize(pred))
    r = Counter(tokenize(ref))
    if not p or not r:
        return 0.0
    return sum((p & r).values()) / max(sum(r.values()), 1)


def sentence_bleu(pred: str, ref: str) -> float:
    """Sentence-level BLEU using sacrebleu (standard implementation)."""
    if not pred or not ref:
        return 0.0
    return sacrebleu.sentence_bleu(pred, [ref]).score / 100.0


def chrf_plusplus(pred: str, ref: str) -> float:
    """chrF++ score using sacrebleu (includes word n-grams, the '++' part)."""
    if not pred or not ref:
        return 0.0
    return sacrebleu.sentence_chrf(pred, [ref], word_order=2).score / 100.0


def keyword_recall(input_words: list[str], pred: str) -> float:
    """Fraction of input content words that appear in the prediction.

    Uses morphology-aware matching via kiwi to avoid false negatives
    from Korean verb/adjective conjugation (e.g., 오다→와서, 아프다→아파).
    """
    function_words = {"나", "너", "우리", "그", "이", "저", "누구", "무엇",
                      "언제", "어디", "어떻게", "왜", "안", "못", "네", "예"}
    keywords = [w for w in input_words if w not in function_words]
    if not keywords:
        return 1.0
    pred_lower = pred.lower()

    # Get morphemes from prediction for lemma-level matching
    pred_morphemes = set()
    try:
        from kiwipiepy import Kiwi
        _kiwi = _get_kiwi()
        result = _kiwi.analyze(pred)
        if result:
            for token in result[0][0]:
                pred_morphemes.add(token.form)
    except Exception:
        pass  # fallback to simple matching if kiwi unavailable

    found = 0
    for kw in keywords:
        kw_lower = kw.lower()
        # 1. Simple substring match
        if kw_lower in pred_lower:
            found += 1
            continue
        # 2. Morpheme-level match (handles conjugation)
        if pred_morphemes:
            # Check the keyword's stem/root
            matched = False
            if kw in pred_morphemes:
                matched = True
            elif kw.endswith("다") and len(kw) > 1:
                stem = kw[:-1]
                if stem in pred_morphemes:
                    matched = True
            # Also analyze the keyword itself to get its root morphemes
            if not matched:
                try:
                    kw_result = _kiwi.analyze(kw)
                    if kw_result:
                        for kt in kw_result[0][0]:
                            if kt.form in pred_morphemes and len(kt.form) > 1:
                                matched = True
                                break
                except Exception:
                    pass
            if matched:
                found += 1
                continue
        # 3. Spacing variation (못자다 → 못 자)
        kw_nospace = kw_lower.replace(" ", "")
        pred_nospace = pred_lower.replace(" ", "")
        if len(kw_nospace) >= 2 and kw_nospace in pred_nospace:
            found += 1
            continue
    return found / len(keywords)


# Singleton kiwi instance for morphological analysis
_kiwi_instance = None

def _get_kiwi():
    global _kiwi_instance
    if _kiwi_instance is None:
        from kiwipiepy import Kiwi
        _kiwi_instance = Kiwi()
    return _kiwi_instance


def negation_accuracy(input_words: list[str], pred: str) -> bool:
    """Check negation preservation."""
    negations = {"없다", "없음", "안", "못", "못하다", "불가능", "불가",
                 "필요없다", "필요 없다", "지 않다", "지 못하다"}
    input_text = " ".join(input_words)
    has_neg = any(nw in input_text for nw in negations)
    if not has_neg:
        return True
    # Check no reversal
    if ("없다" in input_text or "없음" in input_text) and "있다" in pred:
        return False
    if ("못" in input_text or "안" in input_text) and "가능" in pred:
        return False
    return True


def hallucination_rate(input_words: list[str], pred: str) -> float:
    """Fraction of predicted clinical entities NOT found in input.

    Uses kiwi morphological analysis to avoid false positives from
    substring matches (e.g., 간 in 간다, 위 in 위해, 약 in 예약).
    Only counts a clinical term as hallucinated if it appears as an
    independent morpheme in the prediction.
    """
    input_set = set(input_words)
    # Also collect input morpheme stems for broader matching
    input_stems = set()
    for w in input_words:
        input_stems.add(w)
        if w.endswith("다") and len(w) > 1:
            input_stems.add(w[:-1])

    # Use kiwi to get actual morphemes from prediction
    pred_morphemes = set()
    try:
        _kiwi = _get_kiwi()
        result = _kiwi.analyze(pred)
        if result:
            for token in result[0][0]:
                pred_morphemes.add(token.form)
    except Exception:
        # Fallback: simple token split
        pred_morphemes = set(re.sub(r"[\s,.!?。]+", " ", pred).split())

    # Only flag clinical terms that appear as independent morphemes
    hallucinated = []
    for e in MEDICAL_CLINICAL:
        if e in pred_morphemes and e not in input_set and e not in input_stems:
            hallucinated.append(e)

    if not pred_morphemes:
        return 0.0
    return len(hallucinated) / max(len(pred_morphemes), 1) * 100


def medical_entity_recall(input_words: list[str], pred: str) -> float:
    """Recall of medical entities from input in prediction (broad set).

    Uses morphology-aware matching for Korean inflected forms.
    """
    input_entities = [w for w in input_words if w in MEDICAL_BROAD]
    if not input_entities:
        return 1.0
    pred_lower = pred.lower()

    # Get prediction morphemes for lemma matching
    pred_morphemes = set()
    try:
        _kiwi = _get_kiwi()
        result = _kiwi.analyze(pred)
        if result:
            for token in result[0][0]:
                pred_morphemes.add(token.form)
    except Exception:
        pass

    found = 0
    for e in input_entities:
        e_lower = e.lower()
        if e_lower in pred_lower:
            found += 1
        elif e in pred_morphemes:
            found += 1
        elif e.endswith("다") and e[:-1] in pred_morphemes:
            found += 1
        else:
            # Check spacing variation
            if e.replace(" ", "") in pred_lower.replace(" ", "") and len(e) >= 2:
                found += 1
    return found / len(input_entities)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def _morpheme_aware_check(word: str, pred: str, pred_morphemes: set[str]) -> bool:
    """Check if a word is present in prediction using morpheme-level matching."""
    # 1. Simple substring
    if word.lower() in pred.lower():
        return True
    # 2. Morpheme match (stem)
    if word in pred_morphemes:
        return True
    if word.endswith("다") and len(word) > 1:
        stem = word[:-1]
        if stem in pred_morphemes:
            return True
    # 3. Spacing variation
    if word.replace(" ", "") in pred.lower().replace(" ", "") and len(word) >= 2:
        return True
    return False


def classify_errors(input_words: list[str], pred: str, ref: str) -> list[str]:
    """Classify errors into structured categories.

    Uses kiwi morphological analysis for accurate Korean matching.
    """
    errors: list[str] = []
    input_set = set(input_words)
    pred_lower = pred.lower()

    # Get morphemes from prediction for accurate matching
    pred_morphemes = set()
    try:
        _kiwi = _get_kiwi()
        result = _kiwi.analyze(pred)
        if result:
            for token in result[0][0]:
                pred_morphemes.add(token.form)
    except Exception:
        pass

    # Also collect input stems for hallucination check
    input_stems = set()
    for w in input_words:
        input_stems.add(w)
        if w.endswith("다") and len(w) > 1:
            input_stems.add(w[:-1])

    # 1. Symptom omission (morphology-aware)
    symptom_set = {"아프다", "통증", "열", "발열", "기침", "가래", "감기",
                   "어지럽다", "두통", "복통", "설사", "구토", "토하다", "소화",
                   "쓰리다", "더부룩하다", "오한", "떨리다"}
    for s in symptom_set:
        if s in input_set and not _morpheme_aware_check(s, pred, pred_morphemes):
            errors.append("symptom_omission")
            break

    # 2. Body-part omission (morphology-aware)
    bp_set = {"머리", "얼굴", "목", "어깨", "팔", "손", "가슴", "배", "허리",
              "다리", "무릎", "발", "눈", "코", "입", "귀", "혀", "뼈", "피부"}
    for bp in bp_set:
        if bp in input_set and not _morpheme_aware_check(bp, pred, pred_morphemes):
            errors.append("body_part_omission")
            break

    # 3. Test omission (morphology-aware)
    test_set = {"혈액", "혈압", "혈당", "체온", "CT", "MRI", "X-ray", "초음파",
                "내시경", "채혈", "소변"}
    for t in test_set:
        if t in input_set and not _morpheme_aware_check(t, pred, pred_morphemes):
            errors.append("test_omission")
            break

    # 4. Hallucination — morpheme-level only (no substring false positives)
    for e in MEDICAL_CLINICAL:
        if e in pred_morphemes and e not in input_set and e not in input_stems:
            errors.append("hallucination")
            break

    # 5. Negation reversal
    neg_input = {"없다", "없음", "안", "못", "못하다", "불가능"}
    has_neg = any(n in " ".join(input_words) for n in neg_input)
    if has_neg:
        if ("없다" in " ".join(input_words) and "있다" in pred) or \
           ("못" in " ".join(input_words) and "가능" in pred):
            errors.append("negation_reversal")

    # 6. Over-summarization
    if len(pred) < len(ref) * 0.5 and len(input_words) > 2:
        errors.append("over_summarization")
    added = {"매일", "항상", "아침", "저녁", "함께", "정말", "너무", "많이", "모든"}
    for a in added:
        if a in pred_lower and a not in input_set:
            errors.append("over_summarization")
            break

    return errors


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate(data_path: Path, backend: str = "rule", adapter_path: str = "", model_id: str = "", load_4bit: bool = True) -> dict:
    data = json.loads(data_path.read_text(encoding="utf-8"))
    module = SLLMWordsToText(backend=backend, adapter_path=adapter_path, model_id=model_id, load_4bit=load_4bit)
    rows = []

    total_kw_recall = 0.0
    total_neg_acc = 0
    total_neg_count = 0
    total_hal_rate = 0.0
    total_med_recall = 0.0
    total_chrf = 0.0
    error_counts: Counter = Counter()

    for sample in data:
        words = sample["words"]
        reference = sample["reference"]
        pred = module.normalize(words)

        row = {
            "id": sample.get("request_id", str(len(rows) + 1)),
            "words": words,
            "prediction": pred,
            "reference": reference,
            "exact": exact_match(pred, reference),
            "overlap": token_overlap(pred, reference),
            "sentence_bleu": sentence_bleu(pred, reference),
            "chrf_plusplus": chrf_plusplus(pred, reference),
            "keyword_recall": keyword_recall(words, pred),
            "negation_ok": negation_accuracy(words, pred),
            "hallucination_rate": hallucination_rate(words, pred),
            "medical_entity_recall": medical_entity_recall(words, pred),
        }

        # Classify errors
        errors = classify_errors(words, pred, reference)
        row["errors"] = errors
        for e in errors:
            error_counts[e] += 1

        # Accumulate
        total_kw_recall += row["keyword_recall"]
        total_med_recall += row["medical_entity_recall"]
        total_hal_rate += row["hallucination_rate"]
        total_chrf += row["chrf_plusplus"]
        if "없다" in " ".join(words) or "안" in words or "못" in words or "불가능" in words:
            total_neg_count += 1
            if row["negation_ok"]:
                total_neg_acc += 1

        rows.append(row)

    total = len(rows)
    return {
        "total": total,
        "exact_match": sum(r["exact"] for r in rows) / total if total else 0.0,
        "avg_overlap": sum(r["overlap"] for r in rows) / total if total else 0.0,
        "avg_sentence_bleu": sum(r["sentence_bleu"] for r in rows) / total if total else 0.0,
        "avg_chrf_plusplus": total_chrf / total if total else 0.0,
        "avg_keyword_recall": total_kw_recall / total if total else 0.0,
        "negation_accuracy": total_neg_acc / total_neg_count if total_neg_count else 1.0,
        "avg_hallucination_rate": total_hal_rate / total if total else 0.0,
        "avg_medical_entity_recall": total_med_recall / total if total else 0.0,
        "error_analysis": dict(error_counts),
        "rows": rows,
    }


def print_comparison(reports: dict[str, dict]) -> None:
    """Print a comparison table across backends."""
    headers = ["Metric", *reports.keys()]
    metrics = [
        ("Exact Match", "exact_match", "{:.2%}"),
        ("Avg Overlap", "avg_overlap", "{:.2%}"),
        ("BLEU (sacrebleu)", "avg_sentence_bleu", "{:.4f}"),
        ("chrF++", "avg_chrf_plusplus", "{:.4f}"),
        ("Keyword Recall", "avg_keyword_recall", "{:.2%}"),
        ("Negation Accuracy", "negation_accuracy", "{:.2%}"),
        ("Hallucination Rate", "avg_hallucination_rate", "{:.2f}%"),
        ("Medical Entity Recall", "avg_medical_entity_recall", "{:.2%}"),
    ]

    sep = "  |  "
    header_line = " | ".join(headers)
    print(f"\n{'=' * len(header_line)}")
    print("Comparison: Rule vs Exaone vs Finetuned")
    print(f"{'=' * len(header_line)}")
    print(header_line)
    print("-" * len(header_line))

    for label, key, fmt in metrics:
        vals = []
        for name in reports:
            r = reports[name]
            if key == "negation_accuracy":
                v = r.get(key, "N/A")
                vals.append(f"{v:.2%}" if isinstance(v, float) else str(v))
            else:
                v = r.get(key, 0.0)
                vals.append(fmt.format(v))
        row = sep.join([label] + vals)
        print(row)

    # Error analysis comparison
    print(f"\n--- Error Analysis ---")
    all_error_types = set()
    for name, r in reports.items():
        all_error_types.update(r.get("error_analysis", {}).keys())
    sorted_errors = sorted(all_error_types)

    err_header = "Error Type" + sep + sep.join(reports.keys())
    print(err_header)
    print("-" * len(err_header))
    for err_type in sorted_errors:
        vals = [err_type]
        for name in reports:
            cnt = reports[name].get("error_analysis", {}).get(err_type, 0)
            vals.append(str(cnt))
        print(sep.join(vals))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="test_data_clean.json")
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--backend", choices=["rule", "exaone", "finetuned"], default="rule")
    parser.add_argument("--compare", action="store_true",
                        help="Run all 3 backends and print comparison")
    parser.add_argument("--compare-output", default="",
                        help="Dir to save per-backend results (used with --compare)")
    parser.add_argument("--adapter", default="", help="LoRA adapter path (finetuned backend)")
    parser.add_argument("--model-id", default="", help="base model id override (e.g. 7.8B)")
    parser.add_argument("--no-4bit", dest="load_4bit", action="store_false",
                        help="load base in 16bit/bf16 instead of 4bit")
    args = parser.parse_args()

    if args.compare:
        results = {}
        for b in ["rule", "exaone", "finetuned"]:
            print(f"\n--- Evaluating backend: {b} ---")
            report = evaluate(Path(args.data), backend=b)
            results[b] = report
            if args.compare_output:
                out = Path(args.compare_output) / f"evaluation_results_{b}.json"
                out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Saved → {out}")
        print_comparison(results)
    else:
        report = evaluate(Path(args.data), backend=args.backend,
                          adapter_path=args.adapter, model_id=args.model_id, load_4bit=args.load_4bit)
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {k: v for k, v in report.items() if k != "rows"}
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
