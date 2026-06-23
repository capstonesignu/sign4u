"""Quick test: does the improved validator correctly handle supplement-style entries?"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from refilter_all_data import validate_entry

# Cases that SHOULD PASS
good = [
    {"words":["이","아프다","밤","잠","못자다"],"reference":"이가 아파서 밤에 잠을 못 잔다."},
    {"words":["흰색","둥글다","크다","약"],"reference":"흰색이고 둥글고 큰 약이다."},
    {"words":["뼈","부러지다","다리","걷다","못하다"],"reference":"뼈가 부러져서 다리로 걷지 못한다."},
    {"words":["휠체어","타다","병원","가다"],"reference":"휠체어를 타고 병원에 간다."},
    {"words":["수어","통역","필요하다"],"reference":"수어 통역이 필요하다."},
    {"words":["깁스","하다","팔","움직이다","안되다"],"reference":"깁스를 해서 팔을 움직이지 못한다."},
    {"words":["잇몸","붓다","피","나다"],"reference":"잇몸이 부어서 피가 난다."},
    {"words":["귀","안들리다","보청기","끼다"],"reference":"귀가 안 들려서 보청기를 낀다."},
    {"words":["이","빼다","무섭다","참다"],"reference":"이를 빼는 것이 무서워서 참는다."},
    {"words":["허리","아프다","구부리다","안되다"],"reference":"허리가 아파서 구부리지 못한다."},
    {"words":["목발","짚다","걷다","힘들다"],"reference":"목발을 짚고 걷는 것이 힘들다."},
    {"words":["말","이해","못하다","다시","설명하다"],"reference":"말을 이해하지 못해서 다시 설명한다."},
]

# Cases that SHOULD FAIL
bad = [
    {"words":["이","아프다"],"reference":"치통이 심하다."},           # synonym: 이→치통
    {"words":["약","먹다"],"reference":"진통제를 복용한다."},         # synonym: 약→진통제
    {"words":["병원","가다"],"reference":"병원에 갑니다."},           # polite ending
    {"words":["눈","아프다"],"reference":"안구가 아프다."},           # synonym: 눈→안구
    {"words":["배","아프다"],"reference":"복통이 있다."},             # synonym: 배아프다→복통
    {"words":["이","빼다"],"reference":"발치를 한다."},              # synonym: 이빼다→발치
]

print("=== SHOULD PASS (good entries) ===")
all_good = True
for e in good:
    ok, reason = validate_entry(e)
    mark = "OK" if ok else "FAIL"
    if not ok: all_good = False
    words_str = " / ".join(e["words"])
    print(f"  [{mark}] {words_str}")
    if not ok:
        print(f"         reason: {reason}")
        print(f"         ref: {e['reference']}")

print()
print("=== SHOULD FAIL (bad entries) ===")
all_bad = True
for e in bad:
    ok, reason = validate_entry(e)
    mark = "OK" if not ok else "FAIL"
    if ok: all_bad = False
    words_str = " / ".join(e["words"])
    print(f"  [{mark}] {words_str} -> {reason}")

print()
print(f"Good test: {'ALL PASSED' if all_good else 'SOME FAILED'}")
print(f"Bad test:  {'ALL REJECTED' if all_bad else 'SOME LEAKED THROUGH'}")
