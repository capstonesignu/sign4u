"""Prompt templates for the AI-sLLM Words-to-Text module."""

from __future__ import annotations

from typing import Iterable


FEW_SHOT_EXAMPLES = [
    # --- 일상 기본 (5) ---
    (["나", "학교", "오늘", "가다"], "나는 오늘 학교에 갑니다."),
    (["나", "밥", "먹다"], "나는 밥을 먹습니다."),
    (["도움", "필요"], "도움이 필요합니다."),
    (["이해", "못하다", "다시", "설명", "부탁하다"], "이해하지 못해서 다시 설명을 부탁합니다."),
    (["엄마", "병원", "가다"], "엄마가 병원에 갑니다."),
    # --- 의료: 접수/예약 (2) ---
    (["진료", "예약", "하다"], "진료를 예약합니다."),
    (["접수", "하다", "어디", "가다"], "접수를 하려면 어디에 가야 하는지 모릅니다."),
    # --- 의료: 증상 표현 (4) ---
    (["머리", "아프다", "열", "나다"], "머리가 아프고 열이 납니다."),
    (["배", "아프다", "병원", "가다"], "배가 아파서 병원에 갑니다."),
    (["목", "아프다", "기침", "나다"], "목이 아프고 기침이 납니다."),
    (["허리", "아프다", "걷다", "힘들다"], "허리가 아파서 걷기 힘듭니다."),
    # --- 의료: 검사/치료 (3) ---
    (["혈압", "측정", "하다"], "혈압을 측정합니다."),
    (["주사", "맞다", "아프다", "참다"], "주사를 맞고 아프지만 참습니다."),
    (["엑스레이", "찍다", "결과", "기다리다"], "엑스레이를 찍고 결과를 기다립니다."),
    # --- 의료: 약/처방 (2) ---
    (["약", "먹다", "하루", "세", "번"], "약을 하루에 세 번 먹습니다."),
    (["처방전", "받다", "약국", "가다"], "처방전을 받고 약국에 갑니다."),
    # --- 의료: 입원/수술/퇴원 (2) ---
    (["수술", "후", "회복", "중"], "수술 후 회복 중입니다."),
    (["퇴원", "언제", "가능"], "퇴원은 언제 가능한지 묻습니다."),
    # --- 의료: 병원 소통 (2) ---
    (["감기", "걸리다", "쉬다"], "감기에 걸려서 쉽니다."),
    (["알레르기", "있다", "약", "못", "먹다"], "알레르기가 있어서 약을 못 먹습니다."),
]


def format_words(words: Iterable[str]) -> str:
    """Return a stable Korean prompt representation for a word sequence."""
    return " / ".join(str(word).strip() for word in words if str(word).strip())


SYSTEM_INSTRUCTION = (
    "너는 한국수어 번역 시스템의 후처리 sLLM이다.\n"
    "SLR 모델이 인식한 단어 시퀀스를 자연스럽고 문법적으로 올바른 한국어 문장 하나로 변환하라.\n\n"
    "### 필수 규칙 ###\n"
    "1. 종결어미: 반드시 합쇼체(-습니다, -ㅂ니다, -습니까, -ㅂ니까)만 사용한다.\n"
    "   - 금지: -다(평어), -ㄴ다, -는다, -요, -해, -야, -지, -거든\n"
    "   - 올바른 예: 갑니다, 먹습니다, 필요합니다, 좋습니다\n"
    "2. 시제: 입력에 과거 표현(어제, 지난, 했)이 없으면 반드시 현재형으로 쓴다.\n"
    "   - '가다' → '갑니다' (O)  '갔습니다' (X)\n"
    "   - '어제' + '가다' → '갔습니다' (O)\n"
    "3. 정보 추가 금지: 입력에 없는 단어(매일, 아침, 함께, 항상 등)를 절대 추가하지 않는다.\n"
    "4. 출력 형식: 설명이나 부연 없이 문장 하나만 출력한다.\n"
    "5. 조사와 어순만 자연스럽게 보완한다.\n"
)


def build_basic_prompt(words: Iterable[str], domain: str = "daily", context: str = "") -> str:
    word_text = format_words(words)
    context_line = f"\n추가 문맥: {context}" if context else ""
    return (
        SYSTEM_INSTRUCTION
        + f"도메인: {domain}{context_line}\n"
        f"입력 단어: {word_text}\n"
        "출력 문장:"
    )


def build_fewshot_prompt(words: Iterable[str], domain: str = "daily", context: str = "") -> str:
    examples = []
    for example_words, sentence in FEW_SHOT_EXAMPLES:
        examples.append(f"입력 단어: {format_words(example_words)}\n출력 문장: {sentence}")
    word_text = format_words(words)
    context_line = f"\n추가 문맥: {context}" if context else ""
    return (
        SYSTEM_INSTRUCTION
        + "### 예시 ###\n\n"
        + "\n\n".join(examples)
        + "\n\n### 변환 ###\n"
        + f"도메인: {domain}{context_line}\n"
        + f"입력 단어: {word_text}\n"
        + "출력 문장:"
    )


def build_candidate_prompt(candidate_lists, domain: str = "daily", context: str = "") -> str:
    """Task #1 (mechanism B): each position has several candidate words (SLR top-k).

    The model must pick the contextually best word per position and produce ONE
    honorific Korean sentence — no enumeration of combinations.
    `candidate_lists` is a list (per position) of candidate-word lists.
    """
    examples = []
    for example_words, sentence in FEW_SHOT_EXAMPLES[:6]:
        examples.append(f"입력 단어: {format_words(example_words)}\n출력 문장: {sentence}")
    lines = []
    for i, cands in enumerate(candidate_lists, 1):
        cand_text = " / ".join(str(c).strip() for c in cands if str(c).strip())
        lines.append(f"{i}번 위치 후보: {cand_text}")
    cand_block = "\n".join(lines)
    context_line = f"\n추가 문맥: {context}" if context else ""
    return (
        SYSTEM_INSTRUCTION
        + "### 예시 ###\n\n"
        + "\n\n".join(examples)
        + "\n\n### 변환 ###\n"
        + "각 위치마다 여러 후보 단어가 있다(수어 인식 결과). "
        + "각 위치에서 문맥상 가장 적절한 단어 하나씩만 골라, "
        + "자연스럽고 문법적으로 올바른 한국어 문장 하나로 변환하라.\n"
        + f"도메인: {domain}{context_line}\n"
        + cand_block + "\n"
        + "출력 문장:"
    )


def build_candidate_string(candidate_lists) -> str:
    """Compact candidate representation for the fine-tuned model (train==infer).

    [[A, B], [C, D, E]] -> "1) A / B  2) C / D / E"
    Used both when generating candidate training data and at inference, so the
    fine-tuned model sees the exact same format it was trained on.
    """
    parts = []
    for i, cands in enumerate(candidate_lists, 1):
        joined = " / ".join(str(c).strip() for c in cands if str(c).strip())
        parts.append(f"{i}) {joined}")
    return "  ".join(parts)
