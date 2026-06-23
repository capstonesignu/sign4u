"""Diagnose hallucination false positives in detail.

For each of the 26 "hallucination" cases, identify WHICH clinical term
triggered the detection and whether it's a real hallucination or a
substring/morphology false positive.
"""
import json, sys, re
sys.stdout.reconfigure(encoding="utf-8")

MEDICAL_CLINICAL = {
    "아프다", "통증", "열", "발열", "기침", "가래", "감기", "어지럽다",
    "두통", "복통", "설사", "변비", "구토", "토하다", "소화", "쓰리다",
    "더부룩하다", "오한", "떨리다", "부종", "붓다", "발진", "출혈",
    "골절", "염증",
    "혈액", "검사", "소변", "혈압", "혈당", "체온", "CT", "MRI",
    "X-ray", "초음파", "내시경", "채혈", "조영제", "촬영",
    "약", "처방", "처방전", "약국", "진통제", "항생제", "연고",
    "주사", "수액", "알레르기", "부작용", "식전", "식후",
    "위", "장", "간", "심장", "폐", "신장",
}

data = json.loads(open("evaluation_v3_morpho.json", "r", encoding="utf-8").read())

print("=" * 70)
print("HALLUCINATION FALSE POSITIVE DIAGNOSIS")
print("=" * 70)

real_count = 0
false_count = 0

for idx, r in enumerate(data["rows"]):
    if "hallucination" not in r["errors"]:
        continue

    input_set = set(r["words"])
    pred_lower = r["prediction"].lower()

    # Find which terms triggered hallucination
    triggers = []
    for e in MEDICAL_CLINICAL:
        if e in pred_lower and e not in input_set:
            triggers.append(e)

    # Classify each trigger
    print(f"\n[{idx+1}] Input: {' / '.join(r['words'])}")
    print(f"    Pred:  {r['prediction']}")

    is_real = False
    for t in triggers:
        # Check: is this term a substring of a longer word?
        # e.g., "간" in "간다" (go), "간호사" (nurse)
        # e.g., "위" in "위해" (for the purpose of)
        # e.g., "장" in "위장" (stomach), "장애" (disability)
        # e.g., "열" in "열다" (open), "열심히" (hard)

        # Find all occurrences and check context
        false_positive = False
        for match in re.finditer(re.escape(t), pred_lower):
            start, end = match.start(), match.end()
            # Check character before and after
            before = pred_lower[start-1] if start > 0 else " "
            after = pred_lower[end] if end < len(pred_lower) else " "

            # Is it part of a larger Korean word?
            before_is_korean = '가' <= before <= '힣'
            after_is_korean = '가' <= after <= '힣'

            if before_is_korean or after_is_korean:
                false_positive = True
                # Show what word it's actually part of
                # Extract the full word
                ws = start
                while ws > 0 and '가' <= pred_lower[ws-1] <= '힣':
                    ws -= 1
                we = end
                while we < len(pred_lower) and '가' <= pred_lower[we] <= '힣':
                    we += 1
                full_word = pred_lower[ws:we]
                print(f"    FALSE POSITIVE: '{t}' is substring of '{full_word}'")
                break

        if not false_positive:
            # Check if the term was in input but in different form (morphology)
            for iw in input_set:
                if t in iw or iw in t:
                    false_positive = True
                    print(f"    FALSE POSITIVE: '{t}' overlaps with input word '{iw}'")
                    break

        if not false_positive:
            is_real = True
            print(f"    REAL HALLUCINATION: '{t}' not in input, standalone in output")

    if is_real:
        real_count += 1
    else:
        false_count += 1

print(f"\n{'=' * 70}")
print(f"SUMMARY")
print(f"  Total 'hallucination' flags: {real_count + false_count}")
print(f"  False positives:             {false_count}")
print(f"  Real hallucinations:         {real_count}")
print(f"  False positive rate:         {false_count/(real_count+false_count)*100:.0f}%")
print(f"{'=' * 70}")
