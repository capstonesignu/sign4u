"""Honorific (합쇼체) normalization tests for Layer 4 Format Cleanup.

After the honorific patch, FormatCleaner must normalize 평어/해요체 endings
into 합쇼체 (-습니다/-ㅂ니다), and must be idempotent on already-합쇼체 text.
Cases verified empirically against the patched format_cleanup.
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from config import CleanupConfig
from format_cleanup import FormatCleaner


# 평어/해요체 -> 합쇼체 (each value verified against patched code)
PLAIN_TO_POLITE_CASES = [
    ("간다.", "갑니다."),
    ("필요하다.", "필요합니다."),
    ("마셔요.", "마십니다."),
    ("기다린다.", "기다립니다."),
    ("가니?", "갑니까?"),
    ("중이다.", "중입니다."),
    ("좋다.", "좋습니다."),
    ("많다.", "많습니다."),
    ("아프다.", "아픕니다."),
    ("한다.", "합니다."),
    ("온다.", "옵니다."),
    ("먹는다.", "먹습니다."),
]

# already 합쇼체 -> unchanged (idempotency; OLD code would mangle these)
IDEMPOTENT_CASES = ["갑니다.", "필요합니다.", "먹습니다.", "좋습니다."]


@pytest.fixture
def cleaner():
    return FormatCleaner(CleanupConfig())


@pytest.mark.parametrize("plain,polite", PLAIN_TO_POLITE_CASES)
def test_plain_to_polite(cleaner, plain, polite):
    assert cleaner.clean(plain) == polite


@pytest.mark.parametrize("text", IDEMPOTENT_CASES)
def test_idempotent_on_polite(cleaner, text):
    assert cleaner.clean(text) == text
