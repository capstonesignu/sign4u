"""Refilter all data with improved Korean irregular verb handling.

韩语不规则活用类型:
1. ㅂ不规则: 아프다→아파, 덥다→더워, 춥다→추워, 가깝다→가까워
2. ㄷ不规则: 듣다→들어, 걷다→걸어, 묻다→물어
3. ㅅ不规则: 짓다→지어, 낫다→나아
4. ㅎ不规则: 노랗다→노래, 빨갛다→빨개, 하얗다→하얘
5. 르不规则: 모르다→몰라, 빠르다→빨라, 다르다→달라
6. ㅡ脱落: 쓰다→써, 크다→커, 바쁘다→바빠
7. 하다类: 하다→해/한, 공부하다→공부해/공부한
8. ㄹ脱落: 살다→사는, 알다→아는, 만들다→만드는
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 韩语不规则活用词干变体生成
# ============================================================

# ㅂ不规则动词/形容词 (ㅂ → 우/워)
BIUP_IRREGULAR = {
    "아프다": ["아프", "아파", "아픈", "아팠"],
    "덥다": ["덥", "더워", "더운", "더웠"],
    "춥다": ["춥", "추워", "추운", "추웠"],
    "가깝다": ["가깝", "가까워", "가까운", "가까웠"],
    "무겁다": ["무겁", "무거워", "무거운", "무거웠"],
    "쉽다": ["쉽", "쉬워", "쉬운", "쉬웠"],
    "어렵다": ["어렵", "어려워", "어려운", "어려웠"],
    "맵다": ["맵", "매워", "매운", "매웠"],
    "눕다": ["눕", "누워", "누운", "누웠"],
    "줍다": ["줍", "주워", "주운", "주웠"],
    "굽다": ["굽", "구워", "구운", "구웠"],
    "돕다": ["돕", "도와", "도운", "도왔"],
    "곱다": ["곱", "고와", "고운", "고왔"],
    "가볍다": ["가볍", "가벼워", "가벼운", "가벼웠"],
    "부드럽다": ["부드럽", "부드러워", "부드러운"],
    "더럽다": ["더럽", "더러워", "더러운"],
    "뜨겁다": ["뜨겁", "뜨거워", "뜨거운"],
    "차갑다": ["차갑", "차가워", "차가운"],
    "귀엽다": ["귀엽", "귀여워", "귀여운"],
    "무섭다": ["무섭", "무서워", "무서운"],
    "즐겁다": ["즐겁", "즐거워", "즐거운"],
    "슬겁다": ["슬겁", "슬거워"],
    "고맙다": ["고맙", "고마워", "고마운"],
    "반갑다": ["반갑", "반가워", "반가운"],
    "부럽다": ["부럽", "부러워", "부러운"],
    "새롭다": ["새롭", "새로워", "새로운"],
    "싱겁다": ["싱겁", "싱거워", "싱거운"],
}

# ㄷ不规则 (ㄷ → ㄹ)
DIGEUT_IRREGULAR = {
    "듣다": ["듣", "들어", "들은", "들었", "들으"],
    "걷다": ["걷", "걸어", "걸은", "걸었", "걸으"],
    "묻다": ["묻", "물어", "물은", "물었", "물으"],
    "싣다": ["싣", "실어", "실은", "실었"],
    "깨닫다": ["깨닫", "깨달아", "깨달은"],
}

# ㅅ不规则 (ㅅ脱落)
SIOT_IRREGULAR = {
    "짓다": ["짓", "지어", "지은", "지었"],
    "낫다": ["낫", "나아", "나은", "나았"],
    "잇다": ["잇", "이어", "이은", "이었"],
    "붓다": ["붓", "부어", "부은", "부었"],
}

# ㅎ不规则 (ㅎ脱落, 아/어→애/에)
HIEUT_IRREGULAR = {
    "노랗다": ["노랗", "노래", "노란", "노랬"],
    "빨갛다": ["빨갛", "빨개", "빨간", "빨갰"],
    "파랗다": ["파랗", "파래", "파란", "파랬"],
    "하얗다": ["하얗", "하얘", "하얀", "하얬"],
    "까맣다": ["까맣", "까매", "까만", "까맸"],
    "그렇다": ["그렇", "그래", "그런", "그랬"],
    "이렇다": ["이렇", "이래", "이런", "이랬"],
    "저렇다": ["저렇", "저래", "저런", "저랬"],
    "어떻다": ["어떻", "어때", "어떤", "어땠"],
}

# 르不规则
REUR_IRREGULAR = {
    "모르다": ["모르", "몰라", "몰랐"],
    "빠르다": ["빠르", "빨라", "빨랐"],
    "다르다": ["다르", "달라", "달랐"],
    "부르다": ["부르", "불러", "불렀"],
    "자르다": ["자르", "잘라", "잘랐"],
    "마르다": ["마르", "말라", "말랐"],
    "오르다": ["오르", "올라", "올랐"],
    "누르다": ["누르", "눌러", "눌렀"],
    "흐르다": ["흐르", "흘러", "흘렀"],
    "고르다": ["고르", "골라", "골랐"],
    "서두르다": ["서두르", "서둘러", "서둘렀"],
}

# ㅡ脱落
EU_DROP = {
    "쓰다": ["쓰", "써", "썼", "쓴"],
    "크다": ["크", "커", "컸", "큰"],
    "끄다": ["끄", "꺼", "껐", "끈"],
    "뜨다": ["뜨", "떠", "떴", "뜬"],
    "바쁘다": ["바쁘", "바빠", "바빴", "바쁜"],
    "슬프다": ["슬프", "슬퍼", "슬펐", "슬픈"],
    "기쁘다": ["기쁘", "기뻐", "기뻤", "기쁜"],
    "나쁘다": ["나쁘", "나빠", "나빴", "나쁜"],
    "예쁘다": ["예쁘", "예뻐", "예뻤", "예쁜"],
    "배고프다": ["배고프", "배고파", "배고팠", "배고픈"],
    "아프다": ["아프", "아파", "아팠", "아픈"],  # also ㅂ but common
}

# ㄹ脱落 (ㄹ + ㄴ/ㅂ/ㅅ → 脱落ㄹ)
RIEUL_DROP = {
    "살다": ["살", "사는", "사니", "삽니"],
    "알다": ["알", "아는", "아니", "압니"],
    "만들다": ["만들", "만드는", "만드니", "만듭니"],
    "열다": ["열", "여는", "여니"],
    "팔다": ["팔", "파는", "파니"],
    "놀다": ["놀", "노는", "노니"],
    "울다": ["울", "우는", "우니"],
    "풀다": ["풀", "푸는", "푸니"],
}

# 하다 → 해/한
HADA_FORMS = ["해", "한", "했", "하"]

# 되다 → 돼/된
DOEDA_FORMS = ["돼", "된", "됐", "되"]

# 常见动词特殊活用
SPECIAL_VERBS = {
    "가다": ["가", "간", "갔", "갈"],
    "오다": ["오", "온", "왔", "올"],
    "보다": ["보", "본", "봤", "봐", "볼"],
    "주다": ["주", "준", "줬", "줄", "줘"],
    "하다": ["하", "한", "했", "할", "해"],
    "되다": ["되", "된", "됐", "될", "돼"],
    "먹다": ["먹", "먹은", "먹었", "먹을"],
    "마시다": ["마시", "마신", "마셨", "마실", "마셔"],
    "자다": ["자", "잔", "잤", "잘"],
    "사다": ["사", "산", "샀", "살"],
    "타다": ["타", "탄", "탔", "탈"],
    "나다": ["나", "난", "났", "날"],
    "서다": ["서", "선", "섰", "설"],
    "걸리다": ["걸리", "걸린", "걸렸", "걸려", "걸릴"],
    "내리다": ["내리", "내린", "내렸", "내려", "내릴"],
    "올리다": ["올리", "올린", "올렸", "올려", "올릴"],
    "들리다": ["들리", "들린", "들렸", "들려", "들릴"],
    "나오다": ["나오", "나온", "나왔", "나올", "나와"],
    "들어가다": ["들어가", "들어간", "들어갔", "들어갈"],
    "나가다": ["나가", "나간", "나갔", "나갈"],
    "돌아가다": ["돌아가", "돌아간", "돌아갔", "돌아갈"],
    "돌아오다": ["돌아오", "돌아온", "돌아왔", "돌아올"],
    "찾다": ["찾", "찾은", "찾았", "찾을", "찾아"],
    "잡다": ["잡", "잡은", "잡았", "잡을", "잡아"],
    "닫다": ["닫", "닫은", "닫았", "닫을", "닫아"],
    "받다": ["받", "받은", "받았", "받을", "받아"],
    "놓다": ["놓", "놓은", "놓았", "놓을", "놓아", "놔"],
    "넣다": ["넣", "넣은", "넣었", "넣을", "넣어"],
    "좋다": ["좋", "좋은", "좋았", "좋을"],
    "많다": ["많", "많은", "많았", "많을"],
    "적다": ["적", "적은", "적었", "적을"],
    "없다": ["없", "없는", "없었", "없을"],
    "있다": ["있", "있는", "있었", "있을"],
    "같다": ["같", "같은", "같았", "같을"],
    "싫다": ["싫", "싫은", "싫었", "싫어"],
    "좋아하다": ["좋아하", "좋아한", "좋아했", "좋아해"],
    "싶다": ["싶", "싶은", "싶었", "싶어"],
    "데다": ["데", "덴", "뎄", "델"],
    "피우다": ["피우", "피운", "피웠", "피워"],
    # 지다 verbs (지→져)
    "부러지다": ["부러지", "부러져", "부러진", "부러졌"],
    "떨어지다": ["떨어지", "떨어져", "떨어진", "떨어졌"],
    "빠지다": ["빠지", "빠져", "빠진", "빠졌"],
    "나아지다": ["나아지", "나아져", "나아진", "나아졌"],
    "쓰러지다": ["쓰러지", "쓰러져", "쓰러진", "쓰러졌"],
    "무너지다": ["무너지", "무너져", "무너진", "무너졌"],
    # 끼다/뛰다/세다 etc
    "끼다": ["끼", "낀", "꼈", "낄", "껴"],
    "뛰다": ["뛰", "뛴", "뛰었", "뛸", "뛰어"],
    "세다": ["세", "센", "셌", "셀"],
    "데다": ["데", "덴", "뎄", "델", "데어"],
    "켜다": ["켜", "켠", "켰", "켤"],
    "펴다": ["펴", "편", "폈", "펼"],
    "세우다": ["세우", "세운", "세웠", "세워"],
    "구부리다": ["구부리", "구부린", "구부렸", "구부려", "구부릴"],
    "삼키다": ["삼키", "삼킨", "삼켰", "삼켜", "삼킬"],
    "감다": ["감", "감은", "감았", "감아", "감을"],
    "짚다": ["짚", "짚은", "짚었", "짚어", "짚을"],
    "꽂다": ["꽂", "꽂은", "꽂았", "꽂아", "꽂을"],
    "뽑다": ["뽑", "뽑은", "뽑았", "뽑아", "뽑을"],
    "쑤시다": ["쑤시", "쑤신", "쑤셨", "쑤셔"],
    "시리다": ["시리", "시린", "시렸", "시려"],
    "가리키다": ["가리키", "가리킨", "가리켰", "가리켜"],
}

# Build combined lookup
ALL_IRREGULAR = {}
for d in [BIUP_IRREGULAR, DIGEUT_IRREGULAR, SIOT_IRREGULAR, HIEUT_IRREGULAR,
          REUR_IRREGULAR, EU_DROP, RIEUL_DROP, SPECIAL_VERBS]:
    ALL_IRREGULAR.update(d)


def get_all_stems(word: str) -> list:
    """Get all possible stems/forms for a Korean word."""
    forms = set()

    # 1. Original word
    forms.add(word)

    # 2. Check irregular lookup
    if word in ALL_IRREGULAR:
        forms.update(ALL_IRREGULAR[word])

    # 3. Basic stem (remove 다)
    if word.endswith("하다"):
        root = word[:-2]
        forms.add(root)
        forms.add(root + "하")
        forms.add(root + "한")
        forms.add(root + "했")
        forms.add(root + "해")
        forms.add(root + "할")
    elif word.endswith("되다"):
        root = word[:-2]
        forms.add(root)
        forms.add(root + "되")
        forms.add(root + "된")
        forms.add(root + "됐")
        forms.add(root + "돼")
    elif word.endswith("다") and len(word) >= 2:
        stem = word[:-1]
        forms.add(stem)
        # Common suffixes
        if len(stem) >= 1:
            forms.add(stem + "는")
            forms.add(stem + "은")
            forms.add(stem + "을")

    # 4. For single char words, just use as-is
    if len(word) == 1:
        forms.add(word)

    return [f for f in forms if len(f) >= 1]


def word_in_reference(word: str, reference: str) -> bool:
    """Check if a word (or any of its conjugated forms) appears in reference."""
    forms = get_all_stems(word)
    for form in forms:
        if form in reference:
            return True

    # Handle compound negation words: 못자다→"못 자", 안되다→"안 되", 안들리다→"안 들리"
    for prefix in ["못", "안"]:
        if word.startswith(prefix) and len(word) > len(prefix) + 1:
            inner = word[len(prefix):]  # e.g., "자다" from "못자다"
            # Check if prefix + space + any form of inner verb exists
            inner_forms = get_all_stems(inner)
            for f in inner_forms:
                if prefix + " " + f in reference:
                    return True
                # Also check "~지 못하다" pattern for 못X다
                if prefix == "못":
                    for f2 in inner_forms:
                        if f2 + "지 못" in reference:
                            return True

    # Handle 안되다 → "못한다/못하다" equivalence (semantic negation)
    if word == "안되다":
        for neg in ["안 되", "못하", "못한", "못했"]:
            if neg in reference:
                return True

    return False


def validate_entry(entry: dict) -> tuple:
    """Validate with improved Korean irregular verb handling."""
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

    # Check polite endings
    ref_stripped = ref.rstrip(".!?。")
    if any(ref_stripped.endswith(e) for e in ["요", "습니다", "세요", "해요", "시오"]):
        return False, "polite ending"

    # Check each word
    missing = []
    for w in words:
        if not word_in_reference(w, ref):
            missing.append(w)

    if missing:
        return False, f"missing: {missing}"

    return True, "ok"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Refiltering all data with improved Korean verb handling")
    print("=" * 60)

    # 1. Refilter old 5k data
    old_data = json.load(open(BASE_DIR / "train_data_5k.json", "r", encoding="utf-8"))
    old_pass = [e for e in old_data if validate_entry(e)[0]]
    print(f"\nOld 5k: {len(old_data)} → {len(old_pass)} passed ({len(old_pass)/len(old_data)*100:.1f}%)")
    print(f"  Improvement: {len(old_pass) - 2152} more entries recovered (was 2152)")

    # 2. Load already-accepted round data (these all pass by definition)
    round_files = [
        ("Round1", "train_data_v3.json"),
        ("Round2", "train_data_v3_r2.json"),
        ("Round3", "train_data_v3_r3.json"),
        ("Final",  "train_data_v3_final.json"),
    ]

    round_data = {}
    for name, fname in round_files:
        p = BASE_DIR / fname
        if p.exists():
            data = json.load(open(p, "r", encoding="utf-8"))
            round_data[name] = data
            print(f"{name}: {len(data)} entries")

    # 3. Also scan raw batch directories for any saved entries
    batch_dirs = [
        "training_batches_v3",
        "training_batches_v3_r2",
        "training_batches_v3_r3",
        "training_batches_v3_final",
    ]
    raw_entries = []
    for bd in batch_dirs:
        bd_path = BASE_DIR / bd
        if bd_path.exists():
            for f in sorted(bd_path.glob("*.json")):
                if f.name.startswith("_"):
                    continue
                try:
                    entries = json.loads(f.read_text(encoding="utf-8"))
                    raw_entries.extend(entries)
                except:
                    pass
    print(f"\nRaw batch entries found: {len(raw_entries)}")

    # 4. Merge everything with dedup
    refs = set()
    merged = []

    # Priority: new round data first, then old data
    all_sources = []
    for name, data in round_data.items():
        all_sources.append((name, data))
    all_sources.append(("Old5k_improved", old_pass))
    all_sources.append(("RawBatches", raw_entries))

    for source_name, data in all_sources:
        count = 0
        for e in data:
            if not isinstance(e, dict) or "reference" not in e:
                continue
            if e["reference"] not in refs:
                # Re-validate with improved validator
                ok, reason = validate_entry(e)
                if ok:
                    merged.append(e)
                    refs.add(e["reference"])
                    count += 1
        print(f"  Added from {source_name}: {count}")

    print(f"\n{'='*60}")
    print(f"TOTAL MERGED: {len(merged)} entries")
    print(f"{'='*60}")

    # 5. Save merged JSON
    merged_json = BASE_DIR / "train_data_final_merged.json"
    with open(merged_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Saved: {merged_json}")

    # 6. Save chat JSONL for training
    merged_chat = BASE_DIR / "train_data_final_merged_chat.jsonl"
    with open(merged_chat, "w", encoding="utf-8") as f:
        for e in merged:
            chat = {
                "messages": [
                    {"role": "system", "content": "단어 시퀀스를 자연스러운 한국어 문장 하나로 변환하라. 반드시 해라체(-다, -ㄴ다, -는다)만 사용하고, 입력에 없는 단어를 추가하지 마라."},
                    {"role": "user", "content": "입력 단어: " + " / ".join(e["words"])},
                    {"role": "assistant", "content": e["reference"]}
                ]
            }
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")
    print(f"Saved: {merged_chat}")

    # 7. Stats
    word_counts = {}
    for e in merged:
        n = len(e["words"])
        word_counts[n] = word_counts.get(n, 0) + 1
    print(f"\nWord count distribution:")
    for n in sorted(word_counts):
        print(f"  {n} words: {word_counts[n]} entries")

    print(f"\nReady for training!")


if __name__ == "__main__":
    main()
