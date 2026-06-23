"""Test kiwi's morphological analysis capabilities for our use case."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from kiwipiepy import Kiwi

kiwi = Kiwi()

# Test cases from our actual evaluation misses
test_cases = [
    # (input_word, sentence_containing_conjugated_form)
    ("가다", "아빠가 회사에 간다."),
    ("오다", "내일 비가 와서 우산이 필요하다."),
    ("먹다", "나는 라면을 끓여먹는다."),
    ("아프다", "머리가 아파서 약을 먹는다."),
    ("만나다", "친구를 만나 기쁨을 느낀다."),
    ("하다", "나는 도서관에서 공부하기 위해 한다."),
    ("삼키다", "가루약을 물에 타서 삼키기가 어렵다."),
    ("붓다", "수술 부위가 부어오르면 찜질한다."),
    ("걸리다", "감기에 걸려서 쉰다."),
    ("찜질하다", "얼음 찜질한다."),
]

print("=" * 70)
print("KIWI MORPHOLOGICAL ANALYSIS TEST")
print("=" * 70)

for input_word, sentence in test_cases:
    result = kiwi.analyze(sentence)
    morphemes = [(t.form, t.tag) for t in result[0][0]]

    # Check if input_word or its stem can be found
    stem = input_word[:-1] if input_word.endswith("다") else input_word
    found_forms = [f for f, t in morphemes if f == stem or f == input_word]

    print(f"\n  Input word: {input_word} (stem: {stem})")
    print(f"  Sentence:   {sentence}")
    print(f"  Morphemes:  {morphemes}")
    print(f"  Found:      {found_forms if found_forms else 'NOT FOUND'}")

# Now test the problematic hallucination cases
print(f"\n{'=' * 70}")
print("HALLUCINATION FALSE POSITIVE TEST")
print("간(liver) vs 간다(go), 위(stomach) vs 위해(for)")
print("=" * 70)

hal_cases = [
    ("아빠가 회사에 간다.", "간"),      # 간 = liver, but 간다 = go
    ("공부하기 위해 한다.", "위"),        # 위 = stomach, but 위해 = for
    ("예약일이 오늘인지 확인한다.", "약"),  # 약 = medicine, but 예약 = reservation
    ("오늘 시험 때문에 긴장된다.", "장"),   # 장 = intestine, but 긴장 = tension
    ("화장실 가는데 어렵다.", "장"),       # 장 = intestine, but 화장실 = bathroom
    ("간호사를 부른다.", "간"),           # 간 = liver, but 간호사 = nurse
]

for sentence, problem_char in hal_cases:
    result = kiwi.analyze(sentence)
    morphemes = [(t.form, t.tag) for t in result[0][0]]

    # Check: does the problem character appear as an independent morpheme?
    independent = [f for f, t in morphemes if f == problem_char]
    all_containing = [(f, t) for f, t in morphemes if problem_char in f]

    print(f"\n  Sentence:  {sentence}")
    print(f"  Looking for independent '{problem_char}':")
    print(f"  Morphemes: {morphemes}")
    print(f"  Independent '{problem_char}': {independent if independent else 'NO (good!)'}")
    print(f"  In larger morphemes: {all_containing}")
