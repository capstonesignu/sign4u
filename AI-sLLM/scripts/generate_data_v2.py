"""AI-sLLM Training Data Generator V2 — Targeted Weak-Point Data.

Generates training data specifically targeting model weaknesses:
1. Past/present tense disambiguation
2. Causal relationships (A because B)
3. Question sentences
4. Complex medical multi-clause sentences
5. Negation patterns
6. Semantic precision (antonym confusion)

Uses Deepseek API for generation.
"""

from __future__ import annotations

import json
import time
import re
import sys
from pathlib import Path
from openai import OpenAI

API_KEY = ""  # Set your Deepseek API key here
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"
OUTPUT_DIR = Path(__file__).resolve().parent / "training_batches_v2"
PROGRESS_FILE = Path(__file__).resolve().parent / "_progress_v2.json"
FINAL_OUTPUT = Path(__file__).resolve().parent / "train_data_v2.json"

# ---------------------------------------------------------------------------
# Task definitions — each targets a specific weakness
# ---------------------------------------------------------------------------

TASKS = [
    # === 1. 시제 (Tense) — 현재형 only, no past markers ===
    {
        "id": "tense_present_daily",
        "count": 80,
        "prompt": (
            "한국수어 인식 결과(단어 나열)를 자연스러운 한국어 문장으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 일상생활 (학교, 집, 교통, 식사, 취미)\n\n"
            "### 핵심 규칙 ###\n"
            "1. 반드시 현재형 해라체(-다, -ㄴ다, -는다)만 사용\n"
            "2. 과거형(-았다, -었다, -했다) 절대 사용 금지\n"
            "3. '가다'→'간다'(O) '갔다'(X), '먹다'→'먹는다'(O) '먹었다'(X)\n"
            "4. 입력에 '어제','지난' 등 과거 표현이 없으면 무조건 현재형\n"
            "5. words는 2~5개, reference는 한 문장\n\n"
            "80개를 JSON 배열로 출력:\n"
            '[{"words": ["나", "학교", "가다"], "reference": "나는 학교에 간다."},...]'
        ),
    },
    {
        "id": "tense_present_medical",
        "count": 80,
        "prompt": (
            "한국수어 인식 결과를 한국어 문장으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 병원/의료 (진료, 검사, 약, 수술, 입퇴원)\n\n"
            "### 핵심 규칙 ###\n"
            "1. 반드시 현재형 해라체(-다, -ㄴ다, -는다)만 사용\n"
            "2. 과거형 절대 금지: '했다'(X)→'한다'(O), '맞았다'(X)→'맞는다'(O)\n"
            "3. '깁스하다'→'깁스한다'(O) '깁스했다'(X)\n"
            "4. '찜질하다'→'찜질한다'(O) '찜질했다'(X)\n"
            "5. words는 2~6개, reference는 한 문장\n\n"
            "80개를 JSON 배열로 출력:\n"
            '[{"words": ["혈압", "재다", "높다"], "reference": "혈압을 재니 높다."},...]'
        ),
    },
    {
        "id": "tense_past_explicit",
        "count": 40,
        "prompt": (
            "한국수어 인식 결과를 한국어 문장으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 과거 표현이 명시된 문장 (어제, 지난주, 작년 등)\n\n"
            "### 핵심 규칙 ###\n"
            "1. 입력에 반드시 '어제','지난','작년','그때','이전' 등 과거 표현 포함\n"
            "2. 이 경우에만 과거형(-았다, -었다, -했다) 사용 가능\n"
            "3. 해라체만 사용 (-요/-습니다 금지)\n"
            "4. words는 3~6개\n\n"
            "40개를 JSON 배열로 출력:\n"
            '[{"words": ["어제", "병원", "가다"], "reference": "어제 병원에 갔다."},...]'
        ),
    },

    # === 2. 인과관계 (Causal) — A때문에/A해서 B ===
    {
        "id": "causal_medical",
        "count": 80,
        "prompt": (
            "한국수어 인식 결과를 한국어 문장으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 인과관계가 있는 의료 문장\n\n"
            "### 핵심 규칙 ###\n"
            "1. 원인→결과 순서가 올바른 문장 생성\n"
            "2. '-아서/어서', '-아/어', '-고', '-니까' 등 인과 연결어미 활용\n"
            "3. 예: ['배', '아프다', '병원', '가다'] → '배가 아파서 병원에 간다.'(O)\n"
            "   '병원에 가서 배가 아프다'(X) — 인과 역전 금지\n"
            "4. ['목', '마르다', '물', '마시다'] → '목이 말라서 물을 마신다.'(O)\n"
            "   '물을 마시며 목이 마르다'(X)\n"
            "5. 현재형 해라체만 사용, words는 3~6개\n\n"
            "80개를 JSON 배열로 출력:\n"
            '[{"words": ["열", "나다", "병원", "가다"], "reference": "열이 나서 병원에 간다."},...]'
        ),
    },
    {
        "id": "causal_daily",
        "count": 60,
        "prompt": (
            "한국수어 인식 결과를 한국어 문장으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 인과관계가 있는 일상 문장\n\n"
            "### 핵심 규칙 ###\n"
            "1. 원인→결과 순서 올바르게: '피곤하다'+'자다' → '피곤해서 잔다'(O)\n"
            "2. '-아서/어서', '-니까', '-고' 활용\n"
            "3. 현재형 해라체만 사용\n"
            "4. words는 3~5개\n\n"
            "60개를 JSON 배열로 출력:\n"
            '[{"words": ["비", "오다", "우산", "쓰다"], "reference": "비가 와서 우산을 쓴다."},...]'
        ),
    },

    # === 3. 의문문 (Questions) ===
    {
        "id": "question_daily",
        "count": 50,
        "prompt": (
            "한국수어 인식 결과를 한국어 의문문으로 변환하는 학습 데이터를 생성하라.\n\n"
            "### 핵심 규칙 ###\n"
            "1. 입력에 '어디','언제','뭐','누구','왜','어떻게','몇' 등 의문사가 있으면 의문문 출력\n"
            "2. 해라체 의문형: -니?, -ㄴ가?, -는가?, -냐?\n"
            "3. 예: ['너', '어디', '가다'] → '너는 어디에 가니?'\n"
            "4. ['이것', '뭐', '먹다'] → '이것은 뭐를 먹는 건가?'\n"
            "5. words는 2~5개\n\n"
            "50개를 JSON 배열로 출력:\n"
            '[{"words": ["병원", "어디", "있다"], "reference": "병원이 어디에 있니?"},...]'
        ),
    },
    {
        "id": "question_medical",
        "count": 50,
        "prompt": (
            "한국수어 인식 결과를 한국어 의문문으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 병원에서 환자가 묻는 질문\n\n"
            "### 핵심 규칙 ###\n"
            "1. 해라체 의문형(-니?, -ㄴ가?, -는가?) 또는 '-ㄴ지 묻는다/모른다' 형태\n"
            "2. 예: ['퇴원', '언제', '가능'] → '퇴원은 언제 가능한지 묻는다.'\n"
            "3. ['약', '몇', '번', '먹다'] → '약을 몇 번 먹는지 묻는다.'\n"
            "4. words는 3~6개\n\n"
            "50개를 JSON 배열로 출력:\n"
            '[{"words": ["수술", "시간", "얼마", "걸리다"], "reference": "수술 시간이 얼마나 걸리는지 묻는다."},...]'
        ),
    },

    # === 4. 복합 의료 문장 (Complex Medical) ===
    {
        "id": "complex_medical",
        "count": 80,
        "prompt": (
            "한국수어 인식 결과를 한국어 문장으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 복합 의료 문장 (2~3개 절이 연결된 문장)\n\n"
            "### 핵심 규칙 ###\n"
            "1. '-고', '-아서/어서', '-지만', '-면서', '-기 위해' 등으로 절 연결\n"
            "2. 현재형 해라체만 사용\n"
            "3. 입력 단어에 없는 단어 추가 금지\n"
            "4. words는 4~7개\n\n"
            "80개를 JSON 배열로 출력:\n"
            '[{"words": ["혈압", "재다", "수치", "높다", "약", "먹다"], "reference": "혈압을 재니 수치가 높아서 약을 먹는다."},...]'
        ),
    },

    # === 5. 부정문 (Negation) ===
    {
        "id": "negation",
        "count": 50,
        "prompt": (
            "한국수어 인식 결과를 한국어 부정문으로 변환하는 학습 데이터를 생성하라.\n\n"
            "### 핵심 규칙 ###\n"
            "1. '안','못','없다','못하다','불가능' 등 부정 표현 포함\n"
            "2. 부정의 의미 정확히 보존: '못 먹다'→'못 먹는다'(O), '먹는다'(X)\n"
            "3. '없다'→'있다'로 반전 금지\n"
            "4. 현재형 해라체만 사용\n"
            "5. words는 3~6개\n\n"
            "50개를 JSON 배열로 출력:\n"
            '[{"words": ["약", "못", "먹다", "알레르기"], "reference": "알레르기가 있어서 약을 못 먹는다."},...]'
        ),
    },

    # === 6. 의미 정밀도 (Semantic Precision) ===
    {
        "id": "semantic_precision",
        "count": 50,
        "prompt": (
            "한국수어 인식 결과를 한국어 문장으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 반의어/유사어가 혼동되기 쉬운 문장\n\n"
            "### 핵심 규칙 ###\n"
            "1. 입력 단어의 의미를 정확히 반영 (반의어로 바꾸지 말 것)\n"
            "2. '싱겁다'→'싱겁다'(O), '짜다'(X) — 반의어 변환 금지\n"
            "3. '춥다'→'춥다'(O), '덥다'(X)\n"
            "4. '무겁다'→'무겁다'(O), '가볍다'(X)\n"
            "5. 형용사/감각 단어 정확히 보존\n"
            "6. 현재형 해라체만 사용, words는 3~5개\n\n"
            "50개를 JSON 배열로 출력:\n"
            '[{"words": ["음식", "싱겁다", "간", "맞추다"], "reference": "음식이 싱거워서 간을 맞춘다."},...]'
        ),
    },

    # === 7. 수어 특유 표현 (Sign Language Specific) ===
    {
        "id": "sign_language_patterns",
        "count": 60,
        "prompt": (
            "한국수어 인식 결과를 한국어 문장으로 변환하는 학습 데이터를 생성하라.\n"
            "주제: 수어 특유의 어순 패턴 (SOV가 아닌 경우, 토픽-코멘트 구조 등)\n\n"
            "### 핵심 규칙 ###\n"
            "1. 수어는 한국어와 어순이 다를 수 있음\n"
            "2. 단어 순서가 뒤바뀌어도 자연스러운 한국어 문장으로 재구성\n"
            "3. 예: ['가다', '학교', '나'] → '나는 학교에 간다.' (어순 재배열)\n"
            "4. 예: ['아프다', '머리', '약', '먹다'] → '머리가 아파서 약을 먹는다.'\n"
            "5. 현재형 해라체만 사용, words는 3~6개\n\n"
            "60개를 JSON 배열로 출력:\n"
            '[{"words": ["가다", "병원", "아프다", "배"], "reference": "배가 아파서 병원에 간다."},...]'
        ),
    },
]


def extract_json_array(text: str) -> list[dict]:
    """Extract JSON array from possibly messy LLM output."""
    # Try direct parse
    text = text.strip()
    if text.startswith("["):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Find JSON array in text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try to fix truncated JSON
    if "[" in text:
        arr_text = text[text.index("["):]
        # Find last complete object
        last_brace = arr_text.rfind("}")
        if last_brace > 0:
            arr_text = arr_text[:last_brace + 1] + "]"
            try:
                return json.loads(arr_text)
            except json.JSONDecodeError:
                pass

    return []


def validate_entry(entry: dict) -> bool:
    """Validate a single training entry."""
    if not isinstance(entry, dict):
        return False
    words = entry.get("words")
    ref = entry.get("reference")
    if not words or not ref:
        return False
    if not isinstance(words, list) or len(words) < 2:
        return False
    if not isinstance(ref, str) or len(ref) < 3:
        return False
    # Check no polite endings
    polite = ["요.", "습니다.", "세요.", "해요."]
    if any(ref.endswith(p) for p in polite):
        return False
    return True


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    return {"completed": []}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not API_KEY:
        print("ERROR: Set API_KEY in the script first!")
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    OUTPUT_DIR.mkdir(exist_ok=True)

    progress = load_progress()
    all_entries: list[dict] = []
    seen_keys: set[str] = set()

    # Load existing entries
    for task in TASKS:
        batch_file = OUTPUT_DIR / f"{task['id']}.json"
        if batch_file.exists():
            batch = json.loads(batch_file.read_text(encoding="utf-8"))
            for e in batch:
                key = tuple(e["words"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_entries.append(e)

    total_tasks = len(TASKS)
    for i, task in enumerate(TASKS):
        task_id = task["id"]
        if task_id in progress["completed"]:
            print(f"[{i+1}/{total_tasks}] Skipping {task_id} (already done)")
            continue

        print(f"[{i+1}/{total_tasks}] Generating {task_id} ({task['count']} entries)...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "너는 한국수어→한국어 학습 데이터 생성기이다. 요청된 형식의 JSON 배열만 출력하라."},
                    {"role": "user", "content": task["prompt"]},
                ],
                max_tokens=8192,
                temperature=0.8,
            )
            raw = response.choices[0].message.content or ""
            entries = extract_json_array(raw)

            valid = [e for e in entries if validate_entry(e)]
            unique = []
            for e in valid:
                key = tuple(e["words"])
                if key not in seen_keys:
                    seen_keys.add(key)
                    unique.append(e)
                    all_entries.append(e)

            # Save batch
            batch_file = OUTPUT_DIR / f"{task_id}.json"
            batch_file.write_text(json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")

            print(f"  Got {len(entries)} entries, {len(valid)} valid, {len(unique)} unique")

            progress["completed"].append(task_id)
            save_progress(progress)

        except Exception as e:
            print(f"  ERROR: {e}")

        if i < total_tasks - 1:
            print("  Waiting 3s ...")
            time.sleep(3)

    # Save final merged file
    print(f"\n{'='*50}")
    print(f"Total: {len(all_entries)} unique entries")
    FINAL_OUTPUT.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {FINAL_OUTPUT}")

    # Verify
    polite_count = sum(1 for e in all_entries
                       if any(e["reference"].endswith(p) for p in ["요.", "습니다.", "세요."]))
    print(f"Polite endings (should be 0): {polite_count}")
    print("Done!")


if __name__ == "__main__":
    main()
