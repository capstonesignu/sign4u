"""Auto-generate high-quality training data using Deepseek API.

v3 改进点:
1. 强制要求: reference句子必须包含words中的每个词(原文形式或活用形)
2. 禁止同义词替换 (머리아프다→두통 这种不行)
3. 生成后自动验证, 不合格的丢弃
4. 同时输出 train_data_chat.jsonl (训练用) 和 train_data_v3.json (评估用)

用法:
  set DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_API_KEY>
  python generate_data_v3.py
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not API_KEY:
    print("Error: set DEEPSEEK_API_KEY environment variable")
    print("  set DEEPSEEK_API_KEY=sk-your-key-here")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "training_batches_v3"
OUTPUT_DIR.mkdir(exist_ok=True)
MERGED_JSON = BASE_DIR / "train_data_v3.json"
MERGED_CHAT = BASE_DIR / "train_data_v3_chat.jsonl"

# ============================================================
# 核心改进: SYSTEM_MSG 强调"只用输入词原文"
# ============================================================
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

【错误示例 - 绝对不要这样】
{"words":["나","머리","아프다","약","먹다"],"reference":"나는 두통 때문에 진통제를 먹는다."}
→ 错误! "머리아프다"被换成了"두통", "약"被换成了"진통제"
"""

# ============================================================
# 任务列表 (100个子任务, 每个50条 = 5000条)
# ============================================================
SUB_TASKS = [
    # 접수/예약 (10 x 50 = 500)
    ("접수1", "의료 접수/예약 50条：挂号、预约、初诊、复诊、等待叫号。注意words里的每个词必须在reference里出现原文！"),
    ("접수2", "의료 접수/예약 50条：排队、保险卡、身份证、挂号窗口、预约取消"),
    ("접수3", "의료 접수/예약 50条：网上预约、电话预约、预约变更、挂号费、医保"),
    ("접수4", "의료 접수/예약 50条：急诊挂号、转诊、分诊、导诊台、科室选择"),
    ("접수5", "의료 접수/예약 50条：复查预约、术后复诊、定期检查预约"),
    ("접수6", "의료 접수/예약 50条：候诊室、叫号屏幕、排队等候、预约确认短信"),
    ("접수7", "의료 접수/예약 50条：挂号流程说明、自助挂号机、窗口排队"),
    ("접수8", "의료 접수/예약 50条：诊疗卡、就诊记录、病历本、挂号单"),
    ("접수9", "의료 접수/예약 50条：周末门诊、夜间急诊、节假日值班"),
    ("접수10", "의료 접수/예약 50条：陪同就医、代理挂号、老人挂号帮助"),
    # 증상 (16 x 50 = 800)
    ("증상1", "증상 50条：头痛、头晕、偏头痛、眩晕"),
    ("증상2", "증상 50条：发烧、发冷、出汗、体温高低"),
    ("증상3", "증상 50条：咳嗽、痰、气喘、呼吸困难"),
    ("증상4", "증상 50条：鼻塞、流鼻涕、打喷嚏、鼻出血"),
    ("증상5", "증상 50条：喉咙痛、声音哑、吞咽困难"),
    ("증상6", "증상 50条：胃痛、腹泻、便秘、恶心呕吐"),
    ("증상7", "증상 50条：腰疼、关节痛、骨折、扭伤"),
    ("증상8", "증상 50条：皮肤发疹、瘙痒、伤口、烫伤"),
    ("증상9", "증상 50条：眼睛痛、视力模糊、眼干、充血"),
    ("증상10", "증상 50条：耳鸣、听力下降、耳朵痛"),
    ("증상11", "증상 50条：牙疼、牙龈出血、口腔溃疡"),
    ("증상12", "증상 50条：胸闷、心跳快、血压高低"),
    ("증상13", "증상 50条：手脚麻木、肌肉疼痛、抽筋"),
    ("증상14", "증상 50条：失眠、疲劳、体重变化、食欲不振"),
    ("증상15", "증상 50条：尿频、尿痛、小便异常"),
    ("증상16", "증상 50条：过敏反应、肿胀、淋巴结肿大"),
    # 검사/치료 (10 x 50 = 500)
    ("검사1", "검사/치료 50条：抽血、血液检查、血糖血压测量"),
    ("검사2", "검사/치료 50条：X光、CT、MRI拍片"),
    ("검사3", "검사/치료 50条：超声波、心电图、肺功能检查"),
    ("검사4", "검사/치료 50条：内视镜、胃镜、肠镜"),
    ("검사5", "검사/치료 50条：尿检、大便检查、过敏测试"),
    ("검사6", "검사/치료 50条：视力检查、听力检查、口腔检查"),
    ("검사7", "검사/치료 50条：注射、输液、打点滴"),
    ("검사8", "검사/치료 50条：手术缝合、伤口处理、消毒"),
    ("검사9", "검사/치료 50条：物理治疗、康复训练、理疗"),
    ("검사10", "검사/치료 50条：骨密度、甲状腺、组织活检"),
    # 약/처방 (10 x 50 = 500)
    ("약1", "약/처방 50条：口服药、药片、胶囊、粉末药"),
    ("약2", "약/처방 50条：止痛药、消炎药、抗生素"),
    ("약3", "약/처방 50条：过敏药、感冒药、消化药"),
    ("약4", "약/처방 50条：眼药水、外用药膏、贴片"),
    ("약5", "약/처방 50条：处方取药、药房、处方单"),
    ("약6", "약/처방 50条：服药时间（饭前饭后睡前）、剂量"),
    ("약7", "약/처방 50条：药物副作用、过敏反应"),
    ("약8", "약/처방 50条：输液药物、注射药物、胰岛素"),
    ("약9", "약/처방 50条：中药、维生素、保健品"),
    ("약10", "약/처방 50条：换药、停药、加药、药物咨询"),
    # 입원/퇴원 (10 x 50 = 500)
    ("입원1", "입원/수술/퇴원 50条：入院手续、病房分配、住院准备"),
    ("입원2", "입원/수술/퇴원 50条：手术同意书、术前准备、禁食"),
    ("입원3", "입원/수술/퇴원 50条：麻醉、手术过程、术后恢复室"),
    ("입원4", "입원/수술/퇴원 50条：换药、拆线、伤口护理"),
    ("입원5", "입원/수술/퇴원 50条：病房生活、护士呼叫、体温测量"),
    ("입원6", "입원/수술/퇴원 50条：探视、陪护、病房规则"),
    ("입원7", "입원/수술/퇴원 50条：康复训练、下床行走、复健"),
    ("입원8", "입원/수술/퇴원 50条：出院手续、出院教育、注意事项"),
    ("입원9", "입원/수술/퇴원 50条：复查预约、出院后用药"),
    ("입원10", "입원/수술/퇴원 50条：住院费用、保险结算、账单"),
    # 의사소통 (8 x 50 = 400)
    ("소통1", "의사소통 50条：听不懂、再说一次、说慢一点"),
    ("소통2", "의사소통 50条：写字沟通、手机打字、笔谈"),
    ("소통3", "의사소통 50条：需要手语翻译、沟通困难"),
    ("소통4", "의사소통 50条：疼痛程度描述（1-10分）"),
    ("소통5", "의사소통 50条：医嘱确认、注意事项确认"),
    ("소통6", "의사소통 50条：检查结果询问、诊断询问"),
    ("소통7", "의사소통 50条：费用咨询、保险咨询"),
    ("소통8", "의사소통 50条：预约确认、下次就诊时间"),
    # 일상 (12 x 50 = 600)
    ("일상1", "일상 50条：吃饭、做饭、点外卖、食堂"),
    ("일상2", "일상 50条：睡觉、起床、闹钟、失眠"),
    ("일상3", "일상 50条：上学、上课、作业、考试"),
    ("일상4", "일상 50条：上班、下班、加班、出差"),
    ("일상5", "일상 50条：坐公交、地铁、出租车、开车"),
    ("일상6", "일상 50条：购物、超市、网购、付款"),
    ("일상7", "일상 50条：打扫、洗衣服、做家务"),
    ("일상8", "일상 50条：运动、散步、健身、游泳"),
    ("일상9", "일상 50条：天气（热冷雨雪风）"),
    ("일상10", "일상 50条：看手机、上网、看电视、游戏"),
    ("일상11", "일상 50条：洗澡、穿衣、化妆、理发"),
    ("일상12", "일상 50条：旅行、拍照、餐厅、咖啡店"),
    # 감정 (6 x 50 = 300)
    ("감정1", "감정/상태 50条：疼痛、害怕、紧张、焦虑"),
    ("감정2", "감정/상태 50条：开心、感动、感谢、满足"),
    ("감정3", "감정/상태 50条：难过、哭泣、孤独、想家"),
    ("감정4", "감정/상태 50条：生气、烦躁、委屈、压力大"),
    ("감정5", "감정/상태 50条：累、困、饿、渴、无聊"),
    ("감정6", "감정/상태 50条：担心、后悔、尴尬、羞耻"),
    # 가족 (4 x 50 = 200)
    ("가족1", "가족/관계 50条：爸妈、爷爷奶奶、外公外婆"),
    ("가족2", "가족/관계 50条：兄弟姐妹、配偶、子女"),
    ("가족3", "가족/관계 50条：朋友、同事、老师、同学"),
    ("가족4", "가족/관계 50条：医生、护士、药剂师、护工"),
    # 시간/장소 (4 x 50 = 200)
    ("시간1", "시간/장소 50条：今天明天昨天、上午下午晚上"),
    ("시간2", "시간/장소 50条：医院各科室（内科外科眼科耳鼻喉）"),
    ("시간3", "시간/장소 50条：学校、家、公司、商店、公园"),
    ("시간4", "시간/장소 50条：药房、急诊室、检查室、病房、手术室"),
    # 복합 (10 x 50 = 500)
    ("복합1", "복합场景 50条：症状+就医（某处疼→去医院看）"),
    ("복합2", "복합场景 50条：检查+结果（做检查→等结果→听诊断）"),
    ("복합3", "복합场景 50条：用药+副作用（吃药→出现反应）"),
    ("복합4", "복합场景 50条：住院+日常（住院期间吃饭睡觉散步）"),
    ("복합5", "복합场景 50条：家人+医院（家人陪同就医、探视）"),
    ("복합6", "복합场景 50条：时间+就医（什么时候去哪个科）"),
    ("복합7", "복합场景 50条：情感+症状（害怕检查、紧张手术）"),
    ("복합8", "복합场景 50条：沟通+困难（说不清症状、听不懂医嘱）"),
    ("복합9", "복합场景 50条：出院+注意（出院后用药饮食运动）"),
    ("복합10", "복합场景 50条：自由混合，各种场景组合"),
]


# ============================================================
# 韩语动词/形容词词干提取 (用于验证)
# ============================================================
def get_stem(word: str) -> str:
    """提取韩语动词/形容词词干。"""
    if word.endswith("하다"):
        return word[:-2]
    if word.endswith("다") and len(word) >= 2:
        return word[:-1]
    return word


def word_in_reference(word: str, reference: str) -> bool:
    """检查一个词(或其活用形)是否出现在reference中。"""
    if word in reference:
        return True
    stem = get_stem(word)
    if len(stem) >= 2 and stem in reference:
        return True
    if len(word) == 1 and word in reference:
        return True
    return False


def validate_entry(entry: dict) -> tuple:
    """验证一条训练数据的质量。返回 (合格, 原因)。"""
    if not isinstance(entry, dict):
        return False, "not dict"
    if "words" not in entry or "reference" not in entry:
        return False, "missing fields"
    words = entry["words"]
    ref = entry["reference"]
    if not isinstance(words, list) or len(words) < 2:
        return False, "words too short"
    if not ref.strip():
        return False, "empty reference"

    ref_stripped = ref.rstrip(".!?")
    if any(ref_stripped.endswith(e) for e in ["요", "습니다", "세요", "해요", "시오"]):
        return False, "polite ending"

    missing = []
    for w in words:
        if not word_in_reference(w, ref):
            missing.append(w)

    if missing:
        return False, f"missing words: {missing}"

    return True, "ok"


# ============================================================
# API 调用
# ============================================================
def call_deepseek(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8000,
                temperature=0.8,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    [Retry {attempt+1}] {e}")
            time.sleep(5)
    return ""


def parse_json(text: str) -> list:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            chunk = text[start:end + 1]
            last_brace = chunk.rfind("}")
            if last_brace > 0:
                try:
                    return json.loads(chunk[:last_brace + 1] + "]")
                except json.JSONDecodeError:
                    pass
    return []


def to_chat_format(entry: dict) -> dict:
    """转换为 SFT chat 格式 (供 finetune 使用)。"""
    words = entry["words"]
    ref = entry["reference"]
    return {
        "messages": [
            {
                "role": "system",
                "content": "단어 시퀀스를 자연스러운 한국어 문장 하나로 변환하라. "
                           "반드시 해라체(-다, -ㄴ다, -는다)만 사용하고, "
                           "입력에 없는 단어를 추가하지 마라."
            },
            {
                "role": "user",
                "content": f"입력 단어: {' / '.join(words)}"
            },
            {
                "role": "assistant",
                "content": ref
            }
        ]
    }


# ============================================================
# Main
# ============================================================
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    all_data = []
    all_refs = set()
    rejected_total = 0

    progress_file = OUTPUT_DIR / "_progress.json"
    done_tasks = set()
    if progress_file.exists():
        done_tasks = set(json.loads(progress_file.read_text(encoding="utf-8")))

    for f in sorted(OUTPUT_DIR.glob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            entries = json.loads(f.read_text(encoding="utf-8"))
            for e in entries:
                if isinstance(e, dict) and "reference" in e and e["reference"] not in all_refs:
                    all_data.append(e)
                    all_refs.add(e["reference"])
        except Exception:
            pass

    print(f"Existing data: {len(all_data)} entries from previous runs")
    print(f"Tasks done: {len(done_tasks)}/{len(SUB_TASKS)}")
    print(f"Tasks remaining: {len(SUB_TASKS) - len(done_tasks)}")
    print()

    for i, (name, prompt) in enumerate(SUB_TASKS):
        if name in done_tasks:
            continue

        full_prompt = (
            prompt + "\n\n"
            "【再次提醒】reference句子必须包含words里每一个词的原文(或活用形)！"
            "不要用同义词替换！"
        )

        print(f"[{i+1}/{len(SUB_TASKS)}] {name} ...", end=" ", flush=True)
        raw = call_deepseek(full_prompt)
        entries = parse_json(raw)

        unique = []
        rejected = 0
        for e in entries:
            if not isinstance(e, dict):
                continue
            ok, reason = validate_entry(e)
            if not ok:
                rejected += 1
                continue
            if e["reference"] not in all_refs:
                unique.append(e)
                all_refs.add(e["reference"])
                all_data.append(e)

        rejected_total += rejected

        batch_file = OUTPUT_DIR / f"{name}.json"
        batch_file.write_text(json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"accepted {len(unique)}, rejected {rejected}")

        done_tasks.add(name)
        progress_file.write_text(json.dumps(list(done_tasks), ensure_ascii=False), encoding="utf-8")

        time.sleep(1)

    # ---- Final merge ----
    print(f"\n{'='*60}")
    print(f"Total accepted entries: {len(all_data)}")
    print(f"Total rejected entries: {rejected_total}")

    MERGED_JSON.write_text(
        json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved JSON: {MERGED_JSON}")

    with open(MERGED_CHAT, "w", encoding="utf-8") as f:
        for entry in all_data:
            chat = to_chat_format(entry)
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")
    print(f"Saved JSONL (chat format): {MERGED_CHAT}")

    polite = sum(1 for d in all_data
                 if any(d["reference"].rstrip(".!?").endswith(e) for e in ["요", "습니다", "세요", "해요"]))
    print(f"Polite endings (should be 0): {polite}")
    print("Done!")


if __name__ == "__main__":
    main()
