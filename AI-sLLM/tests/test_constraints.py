"""Unit tests for constraints.py — boundary detection and constraint checks."""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deprecated.constraints import (
    _is_independent_word,
    check_keyword_preservation,
    check_negation_preservation,
    check_hallucination,
)


# ---- _is_independent_word ----

def test_boundary_conjugation():
    """간 in 간다 should NOT match (conjugation of 가다, not liver)."""
    assert not _is_independent_word("간", "간다")

def test_boundary_independent_with_particle():
    """간 in '간이 아프다' should match (liver + particle 이)."""
    assert _is_independent_word("간", "간이 아프다")

def test_boundary_compound_word():
    """간 in 간호사 should NOT match."""
    assert not _is_independent_word("간", "간호사를 부른다")

def test_boundary_위해():
    """위 in 위해 should NOT match."""
    assert not _is_independent_word("위", "위해 한다")

def test_boundary_위_stomach():
    """위 in '위가 아프다' should match."""
    assert _is_independent_word("위", "위가 아프다")

def test_boundary_약_reservation():
    """약 in 예약 should NOT match."""
    assert not _is_independent_word("약", "예약한다")

def test_boundary_약_medicine():
    """약 in '약을 먹는다' should match."""
    assert _is_independent_word("약", "약을 먹는다")

def test_boundary_장_bathroom():
    """장 in 화장실 should NOT match."""
    assert not _is_independent_word("장", "화장실 간다")

def test_boundary_열_diligently():
    """열 in 열심히 should NOT match."""
    assert not _is_independent_word("열", "열심히 한다")

def test_boundary_열_fever():
    """열 in '열이 난다' should match."""
    assert _is_independent_word("열", "열이 난다")


# ---- check_keyword_preservation ----

def test_keyword_direct_match():
    # 가다 → stem 가 ≠ 간 (irregular conjugation) — lightweight check can't handle this
    # But 학교 should be found directly
    ok, missing = check_keyword_preservation(["학교", "공부"], "학교에서 공부한다.")
    assert ok, f"Missing: {missing}"

def test_keyword_irregular_verb_limitation():
    """Lightweight stem check can't handle irregular verbs like 가다→간다."""
    ok, missing = check_keyword_preservation(["가다"], "학교에 간다.")
    # This is a known limitation of the lightweight approach.
    # The full kiwi-based check in evaluate.py handles this correctly.
    assert not ok  # Expected: lightweight check misses this

def test_keyword_verb_stem():
    ok, missing = check_keyword_preservation(["먹다"], "밥을 먹는다.")
    assert ok, f"Missing: {missing}"

def test_keyword_missing():
    ok, missing = check_keyword_preservation(["학교", "병원"], "학교에 간다.")
    assert not ok
    assert "병원" in missing


# ---- check_negation_preservation ----

def test_negation_preserved():
    ok, _ = check_negation_preservation(["약", "못", "먹다"], "약을 못 먹는다.")
    assert ok

def test_negation_reversed():
    ok, reason = check_negation_preservation(
        ["없다", "시간"], "시간이 있다."
    )
    assert not ok
    assert "negation" in reason


# ---- check_hallucination ----

def test_hallucination_none():
    ok, items = check_hallucination(["약", "먹다"], "약을 먹는다.")
    assert ok

def test_hallucination_false_positive_간다():
    """간 in 간다 should NOT trigger hallucination."""
    ok, items = check_hallucination(["회사", "가다"], "회사에 간다.")
    assert ok, f"False positive: {items}"

def test_hallucination_false_positive_예약():
    """약 in 예약 should NOT trigger hallucination."""
    ok, items = check_hallucination(["진료", "예약"], "진료를 예약한다.")
    assert ok, f"False positive: {items}"


# ---- Run all tests ----

if __name__ == "__main__":
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
