"""Supplementary generation: Fill gaps in real-world hospital scenarios for deaf/disabled patients.

Target: ~2500 new entries focusing on:
1. 치과 (Dental) - barely 21 entries
2. 정형외과 (Orthopedic) - only 23 entries
3. 장애인편의 (Disability accessibility) - only 14 entries
4. 응급실 (Emergency) - 55 entries, needs more
5. 약국 세부 (Pharmacy details) - 54 entries
6. 실제 진료 흐름 (Real consultation flow) - step-by-step hospital visits
7. 수어 소통 장벽 (Sign language communication barriers)

Uses improved validator from refilter_all_data.py (handles Korean irregular verbs).

Usage:
  set DEEPSEEK_API_KEY=sk-...
  python generate_data_v3_supplement.py
"""
import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# Import improved validator
from refilter_all_data import validate_entry

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    print("Error: set DEEPSEEK_API_KEY environment variable")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "training_batches_supplement"
OUTPUT_DIR.mkdir(exist_ok=True)
MERGED_JSON = BASE_DIR / "train_data_supplement.json"

SYSTEM_MSG = """你是韩国手语翻译训练数据构建专家。生成"单词序列→韩语句子"训练数据对。

【场景背景】聋哑人/残疾人去医院看病，用手语表达后需要翻译成自然韩语。
手语无法拼写专有名词，所以用视觉/动作描述替代。

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

【正确示例 - 手语视觉描述风格】
{"words":["흰색","둥글다","크다","약"],"reference":"흰색이고 둥글고 큰 약이다."}
{"words":["팔","감다","기계","숫자","나오다"],"reference":"팔을 감는 기계에서 숫자가 나온다."}
{"words":["이","아프다","밤","잠","못자다"],"reference":"이가 아파서 밤에 잠을 못 잔다."}
{"words":["뼈","부러지다","다리","걷다","못하다"],"reference":"뼈가 부러져서 다리로 걷지 못한다."}
{"words":["휠체어","타다","병원","가다"],"reference":"휠체어를 타고 병원에 간다."}

【错误示例 - 绝对不要这样】
{"words":["이","아프다"],"reference":"치통이 심하다."} → 错误! "이"被换成了"치통"
"""

SUB_TASKS = [
    # ============================================================
    # 1. 치과 (Dental) — 현재 21개, 목표 +150
    # ============================================================
    ("sup_치과1", "치과-치통 50条: 이/어금니/앞니 + 아프다/쑤시다/시리다 + 밤/낮/먹다+때 조합"),
    ("sup_치과2", "치과-치료 50条: 이+빼다/때우다/씌우다/갈다 + 아프다/무섭다/참다 + 의사/기계"),
    ("sup_치과3", "치과-잇몸 50条: 잇몸+붓다/피나다/아프다 + 양치하다/칫솔/치약 + 관리/조심"),
    ("sup_치과4", "치과-스케일링/교정 50条: 이+깨끗하다/닦다/가지런하다 + 기계+소리+크다+무섭다"),
    ("sup_치과5", "치과-임플란트/틀니 50条: 이+빠지다/없다+새+넣다/만들다 + 불편하다/적응하다"),

    # ============================================================
    # 2. 정형외과 (Orthopedic) — 현재 23개, 목표 +150
    # ============================================================
    ("sup_정형1", "골절-부위 50条: 팔/다리/손목/발목/갈비뼈/손가락 + 부러지다/금가다 + 아프다/붓다"),
    ("sup_정형2", "깁스/부목 50条: 깁스+하다/풀다 + 팔/다리+움직이다+안되다+답답하다+가렵다"),
    ("sup_정형3", "목발/보조기 50条: 목발+짚다+걷다 + 불편하다/힘들다/넘어지다+조심하다"),
    ("sup_정형4", "허리/디스크 50条: 허리+아프다/구부리다+안되다 + 앉다/서다/눕다+힘들다"),
    ("sup_정형5", "재활운동 50条: 팔/다리+천천히+움직이다/구부리다/펴다+아프다+참다+반복하다"),

    # ============================================================
    # 3. 장애인편의 (Disability) — 현재 14개, 목표 +200
    # ============================================================
    ("sup_장애1", "휠체어 이동 50条: 휠체어+타다+밀다+경사로/엘리베이터+찾다+어디+가다"),
    ("sup_장애2", "수어통역 요청 50条: 수어+통역+필요하다/부르다/기다리다 + 의사+말+이해+못하다"),
    ("sup_장애3", "필담 소통 50条: 글+쓰다+종이+보여주다+읽다+이해하다+못하다+다시"),
    ("sup_장애4", "보청기/보조기구 50条: 보청기+끼다/빼다/고장나다/배터리+바꾸다 + 소리+안들리다"),
    ("sup_장애5", "시각장애 50条: 눈+안보이다+안내+필요하다+도와주다+잡다+걷다+조심하다"),
    ("sup_장애6", "장애인 병원접수 50条: 도움+필요하다+글+읽다+못하다+대신+쓰다+설명하다"),
    ("sup_장애7", "보조견/도우미 50条: 개+같이+들어가다+가능/불가능+기다리다+밖+묶다"),
    ("sup_장애8", "장애인 화장실/편의 50条: 화장실+넓다+어디+가다+손잡이+문+열다+닫다"),

    # ============================================================
    # 4. 응급실 (Emergency) — 현재 55개, 목표 +150
    # ============================================================
    ("sup_응급1", "급성통증 50条: 갑자기+아프다+심하다+참다+못하다+쓰러지다+의식+없다"),
    ("sup_응급2", "외상/사고 50条: 넘어지다/부딪히다/베이다/데다 + 피+나다+많다+멈추다+안되다"),
    ("sup_응급3", "의식/호흡 50条: 숨+쉬다+힘들다/안쉬다 + 의식+없다/흐리다 + 빨리+도와주다"),
    ("sup_응급4", "119/구급차 50条: 전화+하다+119+부르다+기다리다+빨리+오다+태우다+가다"),
    ("sup_응급5", "응급실대기 50条: 급하다+아프다+많이+기다리다+언제+보다+순서+먼저"),

    # ============================================================
    # 5. 약국 세부 (Pharmacy) — 현재 54개, 목표 +100
    # ============================================================
    ("sup_약국1", "처방약 받기 50条: 처방전+주다+약사+확인+약+만들다+기다리다+받다"),
    ("sup_약국2", "일반약 구매 50条: 감기약/두통약/소화제/밴드/소독약+사다+어디+있다+얼마"),
    ("sup_약국3", "복약설명 50条: 약사+설명+약+몇번/몇알+아침/저녁+먹다+주의+듣다"),
    ("sup_약국4", "약 구분법 50条: 색깔+모양+크기+약+구분하다+아침약/저녁약+따로+먹다"),

    # ============================================================
    # 6. 실제 진료 흐름 (Step-by-step) — 새로 추가
    # ============================================================
    # 내과 진료
    ("sup_내과1", "감기 진료 50条: 기침+콧물+열+나다+목+아프다+며칠+되다"),
    ("sup_내과2", "소화기 50条: 배+아프다+설사/변비+며칠+먹다+소화+안되다+가스"),
    ("sup_내과3", "만성질환 50条: 혈압+높다/낮다+당뇨+약+매일+먹다+검사+정기+받다"),
    ("sup_내과4", "알레르기 50条: 피부+빨갛다+가렵다+부풀다+먹다+뭐+만지다+뭐+모르다"),
    # 피부과
    ("sup_피부1", "피부증상 50条: 피부+빨갛다/까맣다/하얗다+가렵다/아프다+얼굴/팔/다리"),
    ("sup_피부2", "피부치료 50条: 연고+바르다/약+먹다+햇빛+피하다+씻다+조심하다"),
    # 이비인후과
    ("sup_이비1", "귀 문제 50条: 귀+안들리다/아프다/울리다+소리+이상하다+어지럽다"),
    ("sup_이비2", "코/목 문제 50条: 코+막히다/피나다+목+아프다/쉬다+삼키다+힘들다"),
    # 비뇨기과
    ("sup_비뇨1", "비뇨기 50条: 소변+자주/아프다/피+나다+참다+힘들다+밤+자다+깨다"),

    # ============================================================
    # 7. 수어 소통 장벽 상황 (Sign-specific) — 새로 추가
    # ============================================================
    ("sup_소통1", "이해못함 50条: 말+이해+못하다+다시+천천히+설명하다+그림+보여주다"),
    ("sup_소통2", "통증위치 지목 50条: 여기+아프다+이쪽+저쪽+위/아래/안/밖+가리키다+보다"),
    ("sup_소통3", "시간표현 50条: 어제/오늘/내일/아까/방금/언제+시작+아프다+며칠+되다"),
    ("sup_소통4", "정도표현 50条: 조금/많이/너무/점점/갑자기+아프다/좋아지다/나빠지다"),
    ("sup_소통5", "동의/거부 50条: 네/아니다/괜찮다/싫다/무섭다/이해하다+못하다+다시"),
    ("sup_소통6", "가족연락 50条: 가족+전화+부르다+오다+같이+설명+듣다+걱정+하다"),
    ("sup_소통7", "글쓰기소통 50条: 핸드폰+글+쓰다+보여주다+읽다+답+쓰다+화면+가리키다"),

    # ============================================================
    # 8. 일상생활 보충 (Daily life supplement) — 균형 맞추기
    # ============================================================
    ("sup_일상1", "은행 50条: 돈+넣다/빼다/보내다+카드+통장+번호+적다"),
    ("sup_일상2", "관공서 50条: 서류+발급+신청+기다리다+번호표+뽑다+창구+가다"),
    ("sup_일상3", "마트 50条: 장+보다+카트+담다+무겁다+계산+카드+봉지"),
    ("sup_일상4", "미용실 50条: 머리+자르다+짧다/길다+염색+색+바꾸다+드라이"),
    ("sup_일상5", "우체국 50条: 편지/택배+보내다+주소+쓰다+무겁다+돈+내다+며칠+걸리다"),
]

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

    # Load ALL existing refs to avoid duplicates
    existing_refs = set()
    for fname in ["train_data_final_merged.json", "train_data_v3.json",
                   "train_data_v3_r2.json", "train_data_v3_r3.json",
                   "train_data_v3_final.json", "train_data_5k.json"]:
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

    # Resume support
    progress_file = OUTPUT_DIR / "_progress.json"
    done_tasks = set()
    if progress_file.exists():
        done_tasks = set(json.loads(progress_file.read_text(encoding="utf-8")))

    # Load previous supplement batches
    for f in sorted(OUTPUT_DIR.glob("*.json")):
        if f.name.startswith("_"): continue
        try:
            for e in json.loads(f.read_text(encoding="utf-8")):
                if isinstance(e, dict) and "reference" in e and e["reference"] not in all_refs:
                    all_data.append(e)
                    all_refs.add(e["reference"])
        except: pass

    print(f"Supplement existing: {len(all_data)}")
    print(f"Tasks done: {len(done_tasks)}/{len(SUB_TASKS)}")
    print(f"Tasks remaining: {len(SUB_TASKS) - len(done_tasks)}")
    print()

    for i, (name, prompt) in enumerate(SUB_TASKS):
        if name in done_tasks: continue
        full_prompt = prompt + "\n\n【再次提醒】reference必须包含words每个词的原文(或活用形)！禁止同义词替换！해라체结尾！"
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
        (OUTPUT_DIR / f"{name}.json").write_text(
            json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"accepted {len(unique)}, rejected {rejected}")
        done_tasks.add(name)
        progress_file.write_text(json.dumps(list(done_tasks), ensure_ascii=False), encoding="utf-8")
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"Supplement total accepted: {len(all_data)}")
    print(f"Supplement total rejected: {rejected_total}")
    MERGED_JSON.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {MERGED_JSON}")
    print("Done! Now run merge_final.py to combine with existing data.")


if __name__ == "__main__":
    main()
