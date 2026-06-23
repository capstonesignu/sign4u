"""Layer 4: Format Cleanup.

Pure formatting operations that do NOT change semantics.
Only touches: prefix removal, spacing, sentence clipping,
punctuation, plain→formal-polite (합쇼체) form, token repair.
"""

from __future__ import annotations

import re

from config.cleanup_config import CleanupConfig


# ---------------------------------------------------------------------------
# Prefix patterns (model output artifacts)
# ---------------------------------------------------------------------------

PREFIX_PATTERNS = [
    r"^출력\s*문장\s*[:：]\s*",
    r"^출력\s*[:：]\s*",
    r"^정답\s*[:：]\s*",
    r"^문장\s*[:：]\s*",
    r"^답변\s*[:：]\s*",
]


# ---------------------------------------------------------------------------
# Plain / casual / honorific → formal-polite (합쇼체, -습니다/-ㅂ니다) form
# (formatting normalization, not semantic)
#
# Goal: every sentence ends in 합쇼체. We normalize from three sources:
#   (a) plain 평어   (-ㄴ다/-는다/-다)        e.g. 간다.   → 갑니다.
#   (b) casual 해요체 (-아요/-어요/-해요)       e.g. 가요.   → 갑니다.
#   (c) honorific -세요/-십니다 stays polite but we keep a consistent 합쇼체.
#
# Order matters: longer / more specific patterns first so they win over the
# generic fallbacks (e.g. 합니다 must be handled before the bare 다 fallback).
# ---------------------------------------------------------------------------

PLAIN_TO_POLITE = [
    # --- already 합쇼체: leave as-is (no rule needed) ---

    # --- 해요체 (-아/어/해요) → 합쇼체 ---
    (r"합니다\.?$", "합니다."),               # idempotent guard
    (r"해요\.", "합니다."), (r"해요\?", "합니까?"), (r"해요$", "합니다."),
    (r"돼요\.", "됩니다."), (r"돼요$", "됩니다."),
    (r"가요\.", "갑니다."), (r"가요\?", "갑니까?"), (r"가요$", "갑니다."),
    (r"와요\.", "옵니다."), (r"와요$", "옵니다."),
    (r"봐요\.", "봅니다."), (r"봐요$", "봅니다."),
    (r"줘요\.", "줍니다."), (r"줘요$", "줍니다."),
    (r"마셔요\.", "마십니다."), (r"마셔요$", "마십니다."),
    (r"드셔요\.", "드십니다."), (r"드셔요$", "드십니다."),
    (r"기다려요\.", "기다립니다."), (r"기다려요$", "기다립니다."),
    (r"걸려요\.", "걸립니다."), (r"걸려요$", "걸립니다."),
    (r"느껴요\.", "느낍니다."), (r"느껴요$", "느낍니다."),
    (r"나요\?", "ㅂ니까?"), (r"까요\?", "ㅂ니까?"),
    (r"어요\.", "습니다."), (r"어요\?", "습니까?"), (r"어요$", "습니다."),
    (r"아요\.", "습니다."), (r"아요\?", "습니까?"), (r"아요$", "습니다."),

    # --- casual 반말 (-해/-야/-지) → 합쇼체 ---
    (r"필요해\.?$", "필요합니다."),
    (r"좋아해\.?$", "좋아합니다."),
    (r"싫어해\.?$", "싫어합니다."),

    # --- honorific -세요/-신다 → 합쇼체 (drop subject honorific, keep polite) ---
    (r"가세요\.?$", "갑니다."), (r"오세요\.?$", "옵니다."),
    (r"하세요\.?$", "합니다."), (r"드세요\.?$", "먹습니다."),
    (r"가신다\.?$", "갑니다."), (r"오신다\.?$", "옵니다."),
    (r"하신다\.?$", "합니다."),

    # --- plain 평어 (-ㄴ다/-는다/-다) → 합쇼체 ---
    # verbs in -한다 → -합니다
    (r"한다\.?$", "합니다."), (r"한다\?$", "합니까?"),
    # common irregular verb stems
    (r"간다\.?$", "갑니다."), (r"온다\.?$", "옵니다."),
    (r"먹는다\.?$", "먹습니다."), (r"마신다\.?$", "마십니다."),
    (r"본다\.?$", "봅니다."), (r"잔다\.?$", "잡니다."),
    (r"준다\.?$", "줍니다."), (r"받는다\.?$", "받습니다."),
    (r"듣는다\.?$", "듣습니다."), (r"적는다\.?$", "적습니다."),
    (r"쉰다\.?$", "쉽니다."), (r"참는다\.?$", "참습니다."),
    (r"맞는다\.?$", "맞습니다."), (r"찍는다\.?$", "찍습니다."),
    (r"걸린다\.?$", "걸립니다."), (r"잡는다\.?$", "잡습니다."),
    (r"기다린다\.?$", "기다립니다."), (r"내린다\.?$", "내립니다."),
    (r"오른다\.?$", "오릅니다."), (r"누른다\.?$", "누릅니다."),
    (r"이다\.?$", "입니다."), (r"중이다\.?$", "중입니다."),
    # plain question -니? / -냐? → 합쇼체 question
    (r"가니\?$", "갑니까?"), (r"하니\?$", "합니까?"), (r"오니\?$", "옵니까?"),
    (r"있니\?$", "있습니까?"), (r"없니\?$", "없습니까?"),
    (r"([가-힣])니\?$", r"\1습니까?"),
    (r"된다\.?$", "됩니다."), (r"난다\.?$", "납니다."),
    (r"운다\.?$", "웁니다."), (r"읽는다\.?$", "읽습니다."),
    (r"배운다\.?$", "배웁니다."), (r"산다\.?$", "삽니다."),
    # generic -는다 → -습니다, -ㄴ다 → -ㅂ니다 (after specific stems above)
    (r"는다\.?$", "습니다."),
    (r"ㄴ다\.?$", "ㅂ니다."),

    # adjective / descriptive -다 → -ㅂ니다 / -습니다 (common medical/daily)
    (r"필요하다\.?$", "필요합니다."),
    (r"간단하다\.?$", "간단합니다."),
    (r"불편하다\.?$", "불편합니다."),
    (r"가능하다\.?$", "가능합니다."),
    (r"편리하다\.?$", "편리합니다."),
    (r"간편하다\.?$", "간편합니다."),
    (r"아프다\.?$", "아픕니다."),
    (r"기쁘다\.?$", "기쁩니다."),
    (r"슬프다\.?$", "슬픕니다."),
    (r"바쁘다\.?$", "바쁩니다."),
    (r"힘들다\.?$", "힘듭니다."),
    (r"좋다\.?$", "좋습니다."),
    (r"싫다\.?$", "싫습니다."),
    (r"많다\.?$", "많습니다."),
    (r"적다\.?$", "적습니다."),
    (r"있다\.?$", "있습니다."),
    (r"없다\.?$", "없습니다."),
    (r"긴장된다\.?$", "긴장됩니다."),
    (r"충혈되다\.?$", "충혈됩니다."),
    (r"줄다\.?$", "줍니다."),

    # --- -겠- future plain → 합쇼체 future ---
    (r"가겠다\.?$", "가겠습니다."), (r"자겠다\.?$", "자겠습니다."),
    (r"하겠다\.?$", "하겠습니다."), (r"먹겠다\.?$", "먹겠습니다."),
    (r"마시겠다\.?$", "마시겠습니다."),

    # --- general past-tense plain (-았다/-었다/-였다) → 합쇼체 past (-았습니다…) ---
    # keeps past tense intact, only politens the ending
    (r"았다\.?$", "았습니다."),
    (r"었다\.?$", "었습니다."),
    (r"였다\.?$", "였습니다."),
]


# ---------------------------------------------------------------------------
# Token repair patterns (truncated tokens from tokenizer artifacts)
# ---------------------------------------------------------------------------

TOKEN_REPAIRS = {
    # "오" → "오늘" when 오늘 is in input but truncated in output
    "오늘": ("오 ", "오늘 "),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FormatCleaner:
    """Pure format cleanup — does not change semantics."""

    def __init__(self, config: CleanupConfig | None = None):
        self.config = config or CleanupConfig()

    def clean(self, text: str, input_words: list[str] | None = None) -> str:
        """Apply all format cleanup steps in order."""
        if not text:
            return ""

        result = str(text).strip()

        # 1. Strip code fences
        result = result.replace("```", "").strip()

        # 2. Take first line only
        lines = [line.strip() for line in result.splitlines() if line.strip()]
        if lines:
            result = lines[0]

        # 3. Prefix removal
        if self.config.enable_prefix_removal:
            result = self._remove_prefixes(result)

        # 4. Strip quotes
        result = result.strip("\"'""''")

        # 5. Sentence clipping — keep only first sentence
        if self.config.enable_sentence_clipping:
            result = self._clip_first_sentence(result)

        # 6. Spacing normalization
        if self.config.enable_spacing_fix:
            result = self._fix_spacing(result)

        # 7. Token repair
        if self.config.enable_token_repair and input_words:
            result = self._repair_tokens(result, input_words)

        # 8. Plain/casual → formal-polite (합쇼체) form
        if self.config.enable_plain_to_polite:
            result = self._plain_to_polite(result)

        # 9. Punctuation normalization
        if self.config.enable_punctuation_fix:
            result = self._fix_punctuation(result)

        return result.strip()

    # --- Private methods ---

    def _remove_prefixes(self, text: str) -> str:
        for pattern in PREFIX_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
        return text

    def _clip_first_sentence(self, text: str) -> str:
        match = re.search(r"(.+?[.!?。])", text)
        if match:
            return match.group(1).strip()
        return text

    def _fix_spacing(self, text: str) -> str:
        """Fix Korean spacing issues.

        Only normalizes multiple spaces. Does NOT insert new spaces
        into existing text — that risks breaking valid compound words
        (e.g., 도서관 → 도 서 관). Spacing insertion is too risky
        without a proper Korean tokenizer.
        """
        return re.sub(r"\s+", " ", text).strip()

    def _is_likely_compound(self, particle: str, following: str) -> bool:
        """Check if particle+following is a known compound, not a spacing error."""
        # Common compounds where particle is part of the word
        compounds = {"에서", "까지", "부터", "에게", "로서", "로써"}
        for c in compounds:
            if c.startswith(particle) and following.startswith(c[len(particle):]):
                return True
        return False

    def _repair_tokens(self, text: str, input_words: list[str]) -> str:
        """Repair truncated tokens when the full form is in input."""
        for full_word, (truncated, replacement) in TOKEN_REPAIRS.items():
            if full_word in input_words and full_word not in text and truncated in text:
                text = text.replace(truncated, replacement, 1)
        return text

    def _plain_to_polite(self, text: str) -> str:
        for pattern, replacement in PLAIN_TO_POLITE:
            text = re.sub(pattern, replacement, text)
        return text

    def _fix_punctuation(self, text: str) -> str:
        if text and text[-1] not in ".!?。":
            text += "."
        return text
