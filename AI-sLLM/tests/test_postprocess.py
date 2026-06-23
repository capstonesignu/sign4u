"""Unit tests for postprocess.py — protects against regressions."""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deprecated.postprocess import (
    clean_model_output,
    convert_polite_to_plain,
    convert_past_to_present,
    fix_grammar,
    replace_english_leaks,
    remove_hallucinated_words,
    validate_sentence,
)


# ---- convert_polite_to_plain ----

def test_polite_습니다():
    assert convert_polite_to_plain("간다습니다.") == "간다다."  # edge case
    assert convert_polite_to_plain("합니다.") == "한다."
    assert convert_polite_to_plain("됩니다.") == "된다."

def test_polite_해요():
    assert convert_polite_to_plain("해요.") == "한다."
    assert convert_polite_to_plain("돼요.") == "된다."
    assert convert_polite_to_plain("가요.") == "간다."

def test_polite_세요():
    assert convert_polite_to_plain("가세요.") == "간다."
    assert convert_polite_to_plain("하세요.") == "한다."


# ---- convert_past_to_present ----

def test_past_to_present_no_markers():
    assert convert_past_to_present("했다.") == "한다."
    assert convert_past_to_present("갔다.") == "간다."
    assert convert_past_to_present("먹었다.") == "먹는다."

def test_past_to_present_with_markers():
    # Should NOT convert when input has past-tense markers
    assert convert_past_to_present("했다.", ["어제", "가다"]) == "했다."
    assert convert_past_to_present("갔다.", ["지난", "학교"]) == "갔다."

def test_past_to_present_no_markers_input():
    # Should convert when input has no past markers
    assert convert_past_to_present("했다.", ["나", "공부", "하다"]) == "한다."


# ---- replace_english_leaks ----

def test_english_known_words():
    assert "바늘" in replace_english_leaks("needle을 찔러 채혈한다.")
    assert "얼음" in replace_english_leaks("ice를 바르며 찜질한다.")
    assert "병원" in replace_english_leaks("hospital에 간다.")

def test_english_preserves_abbreviations():
    result = replace_english_leaks("CT를 찍고 MRI 결과를 기다린다.")
    assert "CT" in result
    assert "MRI" in result

def test_english_case_insensitive():
    assert "바늘" in replace_english_leaks("Needle을 사용한다.")
    assert "얼음" in replace_english_leaks("ICE를 바른다.")


# ---- fix_grammar ----

def test_grammar_wrong_conjugation():
    assert fix_grammar("필요한다.") == "필요하다."
    assert fix_grammar("좋는다.") == "좋다."
    assert fix_grammar("없는다.") == "없다."

def test_grammar_truncated_words():
    assert fix_grammar("나는 오 날씨", ["오늘", "날씨"]) == "나는 오늘 날씨"


# ---- remove_hallucinated_words ----

def test_remove_hallucinated():
    result = remove_hallucinated_words("매일 학교에 간다.", ["학교", "가다"])
    assert "매일" not in result
    assert "학교" in result

def test_keep_if_in_input():
    result = remove_hallucinated_words("매일 학교에 간다.", ["매일", "학교", "가다"])
    assert "매일" in result

def test_no_partial_match():
    # "늘" should not be removed from "오늘"
    result = remove_hallucinated_words("오늘 학교에 간다.", ["학교", "가다"])
    assert "오늘" in result


# ---- clean_model_output ----

def test_clean_strips_prefix():
    assert clean_model_output("출력 문장: 나는 학교에 간다.") == "나는 학교에 간다."
    assert clean_model_output("정답: 나는 학교에 간다.") == "나는 학교에 간다."

def test_clean_adds_period():
    result = clean_model_output("나는 학교에 간다")
    assert result.endswith(".")

def test_clean_takes_first_sentence():
    result = clean_model_output("나는 학교에 간다. 이것은 설명입니다.")
    assert result == "나는 학교에 간다."

def test_clean_strips_quotes():
    result = clean_model_output('"나는 학교에 간다."')
    assert result.startswith("나")


# ---- validate_sentence ----

def test_validate_empty():
    ok, _ = validate_sentence("")
    assert not ok

def test_validate_too_long():
    ok, _ = validate_sentence("가" * 121)
    assert not ok

def test_validate_prompt_residue():
    ok, _ = validate_sentence("입력 단어를 문장으로 변환한다.")
    assert not ok

def test_validate_ok():
    ok, _ = validate_sentence("나는 학교에 간다.")
    assert ok


# ---- Run all tests ----

if __name__ == "__main__":
    import inspect
    tests = [(name, obj) for name, obj in globals().items()
             if name.startswith("test_") and callable(obj)]
    passed = 0
    failed = 0
    for name, fn in sorted(tests):
        try:
            fn()
            passed += 1
            print(f"  PASS: {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {name} - {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {name} - {type(e).__name__}: {e}")
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
