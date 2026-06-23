"""Round 3: Generate ~1500 more entries to reach 5000 total.

Need 1468 more. At ~20% pass rate, need ~7500 candidates = 150 batches of 50.
Using 50 tasks to be safe.
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
OUTPUT_DIR = BASE_DIR / "training_batches_v3_r3"
OUTPUT_DIR.mkdir(exist_ok=True)
MERGED_JSON = BASE_DIR / "train_data_v3_r3.json"

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

SUB_TASKS = [
    # 음식/요리 (5)
    ("r3_음식1", "음식/요리 50条：밥/국/반찬/고기/생선/채소 + 먹다/만들다/끓이다/볶다/굽다"),
    ("r3_음식2", "음식/요리 50条：과일/빵/우유/커피/차/물/주스 + 마시다/먹다/사다"),
    ("r3_음식3", "음식/요리 50条：맛있다/맵다/짜다/싱겁다/달다/쓰다 味道描述"),
    ("r3_음식4", "음식/요리 50条：배고프다/배부르다/목마르다/식욕 + 먹다/마시다"),
    ("r3_음식5", "식당/카페 50条：주문하다/시키다/계산하다/포장하다/배달 等"),
    # 집/생활공간 (5)
    ("r3_집1", "집/방 50条：방/거실/부엌/화장실/침대/책상/의자 + 관련동사"),
    ("r3_집2", "집안일 50条：청소하다/빨래하다/설거지/정리하다/쓸다/닦다"),
    ("r3_집3", "집 상태 50条：넓다/좁다/깨끗하다/더럽다/조용하다/시끄럽다"),
    ("r3_집4", "가전제품 50条：에어컨/히터/선풍기/냉장고/세탁기 + 켜다/끄다/쓰다"),
    ("r3_집5", "이사/수리 50条：이사하다/고치다/바꾸다/설치하다/망가지다"),
    # 교통/이동 (5)
    ("r3_교통1", "교통수단 50条：버스/지하철/택시/기차/비행기 + 타다/내리다/갈아타다"),
    ("r3_교통2", "길/방향 50条：오른쪽/왼쪽/앞/뒤/건너편 + 가다/돌다/건너다"),
    ("r3_교통3", "운전 50条：차/운전하다/주차하다/출발하다/도착하다/멈추다"),
    ("r3_교통4", "교통상황 50条：막히다/붐비다/늦다/빠르다/느리다/서두르다"),
    ("r3_교통5", "여행교통 50条：표/예매하다/공항/역/터미널 + 출발/도착"),
    # 쇼핑/돈 (5)
    ("r3_쇼핑1", "쇼핑 50条：사다/팔다/고르다/바꾸다/환불하다/돌려주다"),
    ("r3_쇼핑2", "가격/돈 50条：비싸다/싸다/할인/세일/가격/돈/카드"),
    ("r3_쇼핑3", "옷/신발 50条：입다/벗다/신다/쓰다(모자)/끼다(장갑) + 옷종류"),
    ("r3_쇼핑4", "인터넷쇼핑 50条：주문하다/배송/택배/도착하다/반품하다"),
    ("r3_쇼핑5", "시장/마트 50条：장보다/담다/고르다/무겁다/들다/나르다"),
    # 학교/공부 (5)
    ("r3_학교1", "수업 50条：듣다/배우다/가르치다/질문하다/발표하다/토론하다"),
    ("r3_학교2", "시험/성적 50条：시험/합격/불합격/점수/성적/공부하다/외우다"),
    ("r3_학교3", "숙제/과제 50条：쓰다/읽다/제출하다/완성하다/마치다/시작하다"),
    ("r3_학교4", "학교생활 50条：친구/선배/후배/동아리/급식/체육 等组合"),
    ("r3_학교5", "졸업/진학 50条：졸업하다/입학하다/전공/학과/대학교/장학금"),
    # 건강/운동 (5)
    ("r3_건강1", "운동 50条：달리다/걷다/수영하다/등산하다/자전거타다/스트레칭"),
    ("r3_건강2", "다이어트 50条：살찌다/빠지다/체중/운동하다/식단/조절하다"),
    ("r3_건강3", "수면/휴식 50条：자다/일어나다/낮잠/피곤하다/쉬다/눕다"),
    ("r3_건강4", "위생 50条：손씻다/양치하다/샤워하다/소독하다/마스크/쓰다"),
    ("r3_건강5", "정기검진 50条：건강검진/건강하다/결과/이상없다/주의하다"),
    # 직장/업무 (5)
    ("r3_직장1", "업무 50条：일하다/보고하다/회의하다/결정하다/계획하다/준비하다"),
    ("r3_직장2", "직장관계 50条：상사/부하/동료/협력하다/도와주다/부탁하다"),
    ("r3_직장3", "급여/복지 50条：월급/받다/적다/많다/보너스/휴가/쉬다"),
    ("r3_직장4", "면접/취업 50条：지원하다/면접보다/합격하다/채용/이력서/쓰다"),
    ("r3_직장5", "재택/출장 50条：집/일하다/출장가다/외근/야근/퇴근하다"),
    # 감정 심화 (5)
    ("r3_감정1", "복합감정 50条：기쁘다+놀라다, 슬프다+화나다 等复合情感"),
    ("r3_감정2", "감정+행동 50条：웃다/울다/소리지르다/참다/화내다 + 원인"),
    ("r3_감정3", "걱정/불안 50条：걱정하다/불안하다/초조하다/긴장하다/두렵다"),
    ("r3_감정4", "만족/불만 50条：좋다/싫다/마음에들다/실망하다/기대하다"),
    ("r3_감정5", "그리움/추억 50条：그립다/보고싶다/기억하다/잊다/추억 等"),
    # 기술/디지털 (5)
    ("r3_기술1", "핸드폰 50条：전화하다/문자보내다/충전하다/꺼지다/켜다/울리다"),
    ("r3_기술2", "컴퓨터 50条：켜다/끄다/검색하다/저장하다/삭제하다/다운로드"),
    ("r3_기술3", "인터넷 50条：올리다/찾다/보다/읽다/댓글/사진/영상"),
    ("r3_기술4", "문제/고장 50条：고장나다/안되다/느리다/바이러스/수리하다"),
    ("r3_기술5", "앱/게임 50条：설치하다/삭제하다/업데이트/로그인/비밀번호"),
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
    if not isinstance(entry, dict): return False
    if "words" not in entry or "reference" not in entry: return False
    words, ref = entry["words"], entry["reference"]
    if not isinstance(words, list) or len(words) < 2: return False
    if not ref.strip(): return False
    ref_s = ref.rstrip(".!?")
    if any(ref_s.endswith(e) for e in ["요","습니다","세요","해요","시오"]): return False
    for w in words:
        if not word_in_ref(w, ref): return False
    return True

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

    # Load all existing refs
    existing_refs = set()
    for fname in ["train_data_v3.json", "train_data_v3_r2.json", "train_data_5k.json"]:
        p = BASE_DIR / fname
        if p.exists():
            try:
                for e in json.load(open(p, "r", encoding="utf-8")):
                    if isinstance(e, dict) and "reference" in e:
                        existing_refs.add(e["reference"])
            except: pass
    print(f"Existing refs to skip: {len(existing_refs)}")

    all_data = []
    all_refs = set(existing_refs)
    rejected_total = 0

    progress_file = OUTPUT_DIR / "_progress.json"
    done_tasks = set()
    if progress_file.exists():
        done_tasks = set(json.loads(progress_file.read_text(encoding="utf-8")))

    for f in sorted(OUTPUT_DIR.glob("*.json")):
        if f.name.startswith("_"): continue
        try:
            for e in json.loads(f.read_text(encoding="utf-8")):
                if isinstance(e, dict) and "reference" in e and e["reference"] not in all_refs:
                    all_data.append(e)
                    all_refs.add(e["reference"])
        except: pass

    print(f"Round 3 existing: {len(all_data)}")
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
            if not validate_entry(e): rejected += 1; continue
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
    print(f"Round 3 total accepted: {len(all_data)}")
    print(f"Round 3 total rejected: {rejected_total}")
    MERGED_JSON.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {MERGED_JSON}")
    print("Done!")

if __name__ == "__main__":
    main()
