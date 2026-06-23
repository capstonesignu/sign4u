"""Round 2: Generate 2000 more training entries with different prompts.

Reuses the same validation logic from v3, but with different sub-task angles
to avoid duplicating round 1 data.

用法:
  set DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
  python generate_data_v3_round2.py
"""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    print("Error: set DEEPSEEK_API_KEY environment variable")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "training_batches_v3_r2"
OUTPUT_DIR.mkdir(exist_ok=True)
MERGED_JSON = BASE_DIR / "train_data_v3_r2.json"

SYSTEM_MSG = """你是韩国手语翻译训练数据构建专家。生成"单词序列→韩语句子"训练数据对。

【最重要的规则 - 必须严格遵守】
1. reference句子中必须使用words里的每一个词的【原始形态】
   - words有"머리" → reference必须出现"머리", 不能换成"두통"或"뇌"
   - words有"약" → reference必须出现"약", 不能换成"진통제"或"소염제"
   - words有"아프다" → reference可以活用为"아파서/아프고/아픈", 但词干"아프"必须保留
   - words有"먹다" → reference可以活用为"먹는다/먹고/먹어서", 但词干"먹"必须保留

2. reference中【禁止添加】words里没有的实义名词/动词/形容词
   - 可以添加: 助词(은/는/이/가/을/를/에/에서/으로/과/와/도)
   - 可以添加: 连接词尾(-고/-아서/-면/-지만)
   - 禁止添加: words里没有的名词、动词、形容词

3. 해라체结尾（-다,-ㄴ다,-는다）。禁止-요/-습니다/-세요

4. words用动词/形容词原型, 2~6个词

5. 不重复

输出严格JSON数组, 无解释:
[{"words":["단어1","단어2","단어3"],"reference":"문장."},...]

【正确示例】
{"words":["나","머리","아프다","약","먹다"],"reference":"나는 머리가 아파서 약을 먹는다."}
{"words":["배","아프다","병원","가다"],"reference":"배가 아파서 병원에 간다."}
{"words":["오늘","날씨","춥다","옷","입다"],"reference":"오늘 날씨가 추워서 옷을 입는다."}

【错误示例 - 绝对不要这样】
{"words":["나","머리","아프다","약","먹다"],"reference":"나는 두통 때문에 진통제를 먹는다."}
→ 错误! "머리아프다"被换成了"두통", "약"被换成了"진통제"
"""

# Round 2: 40 tasks x 50 = 2000 target (different angles from round 1)
SUB_TASKS = [
    # 신체부위+증상 조합 (8 x 50 = 400)
    ("r2_신체1", "신체+증상 조합 50条：머리/이마/관자놀이 + 아프다/어지럽다/무겁다 等组合"),
    ("r2_신체2", "신체+증상 조합 50条：목/어깨/등 + 아프다/결리다/뻣뻣하다 等组合"),
    ("r2_신체3", "신체+증상 조합 50条：팔/손/손가락 + 저리다/붓다/아프다 等组合"),
    ("r2_신체4", "신체+증상 조합 50条：다리/무릎/발 + 아프다/붓다/저리다 等组合"),
    ("r2_신체5", "신체+증상 조합 50条：눈/코/귀/입 + 아프다/막히다/안보이다 等组合"),
    ("r2_신체6", "신체+증상 조합 50条：가슴/배/허리 + 아프다/답답하다/쓰리다 等组합"),
    ("r2_신체7", "신체+증상 조합 50条：피부/얼굴/손 + 가렵다/붓다/빨갛다 等组合"),
    ("r2_신체8", "신체+증상 조합 50条：전신 증상 - 열나다/춥다/떨리다/땀나다 等组합"),
    # 동사 다양화 (6 x 50 = 300)
    ("r2_동사1", "이동/방향 동사 50条：가다/오다/나가다/들어가다/돌아가다/걸어가다 等组合"),
    ("r2_동사2", "생활 동사 50条：씻다/닦다/빨다/정리하다/치우다/버리다 等组合"),
    ("r2_동사3", "소통 동사 50条：말하다/듣다/묻다/답하다/설명하다/알리다 等组合"),
    ("r2_동사4", "상태변화 동사 50条：되다/변하다/낫다/좋아지다/나빠지다 等组合"),
    ("r2_동사5", "수수 동사 50条：주다/받다/빌리다/돌려주다/보내다/가져오다 等组합"),
    ("r2_동사6", "감각 동사 50条：보다/듣다/느끼다/맡다/만지다/맛보다 等组合"),
    # 부정문/의문문 (4 x 50 = 200)
    ("r2_부정1", "부정문 50条：안/못/없다/모르다 - 不能做某事的句子"),
    ("r2_부정2", "부정문 50条：-지 않다/-지 못하다 - 否定活用形"),
    ("r2_부정3", "금지/경고 50条：하지 말다/조심하다/위험하다/주의하다"),
    ("r2_부정4", "부족/결핍 50条：부족하다/모자라다/없다/필요하다"),
    # 시간순서 (4 x 50 = 200)
    ("r2_시간1", "과거 사건 50条：어제/지난주/아까 + 했다/갔다/먹었다 过去式"),
    ("r2_시간2", "미래 계획 50条：내일/다음주/나중에 + 하겠다/할것이다 将来"),
    ("r2_시간3", "순서/연속 50条：먼저/그다음/마지막 + 动作连续"),
    ("r2_시간4", "빈도/습관 50条：매일/자주/가끔/항상 + 日常习惯"),
    # 원인-결과 (4 x 50 = 200)
    ("r2_인과1", "원인-결과 50条：아프다→병원가다, 배고프다→먹다 类因果"),
    ("r2_인과2", "원인-결과 50条：비오다→우산쓰다, 춥다→옷입다 类因果"),
    ("r2_인과3", "조건문 50条：-면/-으면 条件句型"),
    ("r2_인과4", "양보문 50条：-지만/-아도 转折/让步句型"),
    # 2词/3词 짧은 문장 (4 x 50 = 200)
    ("r2_짧은1", "2词短句 50条：주어+동사 最简单的组合 (나/가다, 비/오다 等)"),
    ("r2_짧은2", "2词短句 50条：명사+형용사 (날씨/좋다, 음식/맛있다 等)"),
    ("r2_짧은3", "3词短句 50条：주어+목적어+동사 (나/밥/먹다 等)"),
    ("r2_짧은4", "3词短句 50条：장소+동사+상태 (학교/가다/싫다 等)"),
    # 5~6词 복잡한 문장 (4 x 50 = 200)
    ("r2_긴1", "5-6词复句 50条：증상描述+处置 (어디/아프다/어떻게/하다/병원/가다)"),
    ("r2_긴2", "5-6词复句 50条：日常叙事 (누가/어디서/무엇을/왜/하다)"),
    ("r2_긴3", "5-6词复句 50条：医院场景复杂句"),
    ("r2_긴4", "5-6词复句 50条：情感+原因+行动 三层结构"),
    # 학교/직장 (3 x 50 = 150)
    ("r2_학교1", "학교 생활 50条：수업/숙제/시험/선생님/교실 等学校场景"),
    ("r2_학교2", "직장 생활 50条：회의/보고서/상사/동료/출퇴근 等职场场景"),
    ("r2_학교3", "사회생활 50条：은행/우체국/관공서/마트/식당 等社会场景"),
    # 날씨/계절/자연 (3 x 50 = 150)
    ("r2_자연1", "날씨 50条：맑다/흐리다/비/눈/바람/덥다/춥다 天气相关"),
    ("r2_자연2", "계절 50条：봄/여름/가을/겨울 + 相关活动"),
    ("r2_자연3", "자연/환경 50条：산/바다/공원/나무/꽃 自然环境"),
]


def get_stem(word):
    if word.endswith("하다"): return word[:-2]
    if word.endswith("다") and len(word) >= 2: return word[:-1]
    return word

def word_in_ref(word, ref):
    if word in ref: return True
    stem = get_stem(word)
    if len(stem) >= 2 and stem in ref: return True
    if len(word) == 1 and word in ref: return True
    return False

def validate_entry(entry):
    if not isinstance(entry, dict): return False, "not dict"
    if "words" not in entry or "reference" not in entry: return False, "missing"
    words, ref = entry["words"], entry["reference"]
    if not isinstance(words, list) or len(words) < 2: return False, "short"
    if not ref.strip(): return False, "empty"
    ref_s = ref.rstrip(".!?")
    if any(ref_s.endswith(e) for e in ["요","습니다","세요","해요","시오"]):
        return False, "polite"
    missing = [w for w in words if not word_in_ref(w, ref)]
    if missing: return False, f"missing: {missing}"
    return True, "ok"

def call_deepseek(prompt, retries=3):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8000, temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [Retry {attempt+1}] {e}")
            time.sleep(5)
    return ""

def parse_json(text):
    try: return json.loads(text)
    except: pass
    s, e = text.find("["), text.rfind("]")
    if s != -1 and e != -1:
        try: return json.loads(text[s:e+1])
        except:
            lb = text[s:e+1].rfind("}")
            if lb > 0:
                try: return json.loads(text[s:lb+1] + "]")
                except: pass
    return []


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Load existing round 1 refs to avoid duplicates
    existing_refs = set()
    for f in [BASE_DIR / "train_data_v3.json", BASE_DIR / "train_data_5k.json"]:
        if f.exists():
            try:
                for e in json.load(open(f, "r", encoding="utf-8")):
                    if isinstance(e, dict) and "reference" in e:
                        existing_refs.add(e["reference"])
            except: pass
    print(f"Existing references to skip: {len(existing_refs)}")

    all_data = []
    all_refs = set(existing_refs)
    rejected_total = 0

    progress_file = OUTPUT_DIR / "_progress.json"
    done_tasks = set()
    if progress_file.exists():
        done_tasks = set(json.loads(progress_file.read_text(encoding="utf-8")))

    # Load previous round 2 batches
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        if f.name.startswith("_"): continue
        try:
            for e in json.loads(f.read_text(encoding="utf-8")):
                if isinstance(e, dict) and "reference" in e and e["reference"] not in all_refs:
                    all_data.append(e)
                    all_refs.add(e["reference"])
        except: pass

    print(f"Round 2 existing: {len(all_data)} entries")
    print(f"Tasks done: {len(done_tasks)}/{len(SUB_TASKS)}")
    print()

    for i, (name, prompt) in enumerate(SUB_TASKS):
        if name in done_tasks: continue
        full_prompt = prompt + "\n\n【再次提醒】reference必须包含words每个词的原文(或活用形)！禁止同义词替换！"
        print(f"[{i+1}/{len(SUB_TASKS)}] {name} ...", end=" ", flush=True)
        raw = call_deepseek(full_prompt)
        entries = parse_json(raw)
        unique, rejected = [], 0
        for e in entries:
            if not isinstance(e, dict): continue
            ok, reason = validate_entry(e)
            if not ok: rejected += 1; continue
            if e["reference"] not in all_refs:
                unique.append(e)
                all_refs.add(e["reference"])
                all_data.append(e)
        rejected_total += rejected
        (OUTPUT_DIR / f"{name}.json").write_text(json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"accepted {len(unique)}, rejected {rejected}")
        done_tasks.add(name)
        progress_file.write_text(json.dumps(list(done_tasks), ensure_ascii=False), encoding="utf-8")
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Round 2 total accepted: {len(all_data)}")
    print(f"Round 2 total rejected: {rejected_total}")
    MERGED_JSON.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {MERGED_JSON}")
    print("Done!")

if __name__ == "__main__":
    main()
