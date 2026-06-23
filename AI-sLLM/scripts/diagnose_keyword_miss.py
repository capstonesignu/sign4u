"""Diagnose keyword recall misses: are they real or evaluator errors?

Uses kiwi morphological analyzer to do lemma-level comparison.
Categorizes each "miss" into:
  - morphology: word present in conjugated/inflected form (evaluator false negative)
  - synonym: semantically equivalent but different word
  - substring: word present as part of a compound word
  - spacing: word present but spacing differs
  - real_miss: genuinely missing from output
"""
import json
import sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

from kiwipiepy import Kiwi

kiwi = Kiwi()


def get_lemmas(text: str) -> set[str]:
    """Extract all lemmas from text using kiwi morphological analysis."""
    result = kiwi.analyze(text)
    lemmas = set()
    if result:
        for token in result[0][0]:
            lemmas.add(token.form)  # surface form of morpheme
            # Also add the lemma if available
            lemmas.add(token.form)
    return lemmas


def get_morphemes(text: str) -> list[tuple[str, str]]:
    """Get (form, tag) pairs for all morphemes in text."""
    result = kiwi.analyze(text)
    morphemes = []
    if result:
        for token in result[0][0]:
            morphemes.append((token.form, token.tag))
    return morphemes


def classify_miss(word: str, prediction: str, pred_lemmas: set[str],
                  pred_morphemes: list[tuple[str, str]]) -> str:
    """Classify why a keyword was 'missed'."""
    word_lower = word.lower()
    pred_lower = prediction.lower()

    # 1. Exact match (shouldn't be here, but safety check)
    if word_lower in pred_lower:
        return "not_actually_missing"

    # 2. Morphology: check if the word's lemma appears in prediction's lemmas
    # Get lemmas for the input word
    word_lemmas = get_lemmas(word)
    # Check verb stems: 아프다 → 아프, 아파, 아픈, 아팠...
    if word.endswith("다") and len(word) > 1:
        stem = word[:-1]
        if stem in pred_lower:
            return "morphology"
        # Check if stem appears as any morpheme
        for form, tag in pred_morphemes:
            if form == stem or form == word:
                return "morphology"

    # Check all lemma overlaps
    for wl in word_lemmas:
        if wl in pred_lemmas and len(wl) > 1:
            return "morphology"

    # 3. Substring: compound word contains the keyword
    # e.g., 찜질 in 온찜질, 약 in 약국
    if len(word_lower) >= 2:
        # Check if word is part of any token in prediction
        pred_tokens = pred_lower.replace(".", "").replace(",", "").split()
        for pt in pred_tokens:
            if word_lower in pt and word_lower != pt:
                return "substring"

    # 4. Spacing: word might be split differently
    # e.g., "못자다" → "못 자다" or "안되다" → "안 되다"
    word_nospace = word_lower.replace(" ", "")
    pred_nospace = pred_lower.replace(" ", "")
    if word_nospace in pred_nospace:
        return "spacing"

    # 5. Check kiwi analysis of input word vs prediction morphemes
    word_morphs = get_morphemes(word)
    for wform, wtag in word_morphs:
        if len(wform) >= 2:
            for pform, ptag in pred_morphemes:
                if wform == pform:
                    return "morphology"

    # 6. Real miss
    return "real_miss"


def main():
    # Load test data and evaluation results
    test_data = json.loads(Path("test_data.json").read_text(encoding="utf-8"))
    eval_results = json.loads(Path("evaluation_v3.json").read_text(encoding="utf-8"))

    function_words = {"나", "너", "우리", "그", "이", "저", "누구", "무엇",
                      "언제", "어디", "어떻게", "왜", "안", "못", "네", "예"}

    miss_categories = Counter()
    all_misses = []
    total_keywords = 0
    total_found = 0

    for row in eval_results["rows"]:
        words = row["words"]
        pred = row["prediction"]
        ref = row["reference"]

        # Get morphological analysis of prediction
        pred_lemmas = get_lemmas(pred)
        pred_morphemes = get_morphemes(pred)

        keywords = [w for w in words if w not in function_words]
        total_keywords += len(keywords)

        for kw in keywords:
            if kw.lower() in pred.lower():
                total_found += 1
                continue

            # This keyword was "missed" by the simple evaluator
            category = classify_miss(kw, pred, pred_lemmas, pred_morphemes)
            miss_categories[category] += 1

            if category != "not_actually_missing":
                total_found_morpho = 1 if category in ("morphology", "substring", "spacing") else 0
                all_misses.append({
                    "input_word": kw,
                    "category": category,
                    "prediction": pred[:80],
                    "reference": ref[:80],
                })

    # Summary
    print("=" * 70)
    print("KEYWORD MISS DIAGNOSIS (kiwi morphology-aware)")
    print("=" * 70)

    print(f"\nTotal input keywords: {total_keywords}")
    print(f"Found by simple match: {total_found} ({total_found/total_keywords*100:.1f}%)")

    total_misses = sum(miss_categories.values())
    print(f"Total 'misses': {total_misses}")
    print()

    # Category breakdown
    print("Miss categories:")
    print("-" * 50)
    false_negatives = 0
    for cat, count in miss_categories.most_common():
        is_false = cat in ("morphology", "substring", "spacing", "not_actually_missing")
        marker = " ← FALSE NEGATIVE" if is_false else ""
        print(f"  {cat:<25s}: {count:>3d} ({count/total_misses*100:.1f}%){marker}")
        if is_false:
            false_negatives += count

    print(f"\n  False negatives (evaluator error): {false_negatives}")
    print(f"  Real misses:                       {total_misses - false_negatives}")

    # Corrected keyword recall
    corrected_found = total_found + false_negatives
    print(f"\n{'=' * 70}")
    print(f"CORRECTED KEYWORD RECALL")
    print(f"  Original (simple match):    {total_found}/{total_keywords} = {total_found/total_keywords*100:.1f}%")
    print(f"  Corrected (morpho-aware):   {corrected_found}/{total_keywords} = {corrected_found/total_keywords*100:.1f}%")
    print(f"  Improvement:                +{(corrected_found-total_found)/total_keywords*100:.1f}%")
    print(f"{'=' * 70}")

    # Show examples of each category
    print(f"\n{'=' * 70}")
    print("EXAMPLES BY CATEGORY")
    print(f"{'=' * 70}")

    for cat in ["morphology", "substring", "spacing", "real_miss"]:
        examples = [m for m in all_misses if m["category"] == cat][:5]
        if examples:
            print(f"\n[{cat}] ({len([m for m in all_misses if m['category'] == cat])} total)")
            for ex in examples:
                print(f"  input_word: {ex['input_word']}")
                print(f"  prediction: {ex['prediction']}")
                print()


if __name__ == "__main__":
    main()
