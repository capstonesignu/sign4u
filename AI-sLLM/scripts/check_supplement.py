"""Check actual supplement data breakdown and verify final training data."""
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent

# 1. Check supplement batches by category
print("=" * 60)
print("补充数据 各分类实际统计")
print("=" * 60)

batch_dir = BASE / "training_batches_supplement"
categories = {
    "치과 (牙科)": "sup_치과",
    "정형외과 (骨科)": "sup_정형",
    "장애인편의 (残疾人)": "sup_장애",
    "응급실 (急诊)": "sup_응급",
    "약국 (药房)": "sup_약국",
    "내과 (内科)": "sup_내과",
    "피부과 (皮肤科)": "sup_피부",
    "이비인후과 (耳鼻喉)": "sup_이비",
    "비뇨기과 (泌尿科)": "sup_비뇨",
    "수어소통 (手语沟通)": "sup_소통",
    "일상생활 (日常)": "sup_일상",
}

total_sup = 0
for label, prefix in categories.items():
    count = 0
    for f in sorted(batch_dir.glob(f"{prefix}*.json")):
        if f.name.startswith("_"): continue
        try:
            entries = json.loads(f.read_text(encoding="utf-8"))
            count += len(entries)
        except: pass
    total_sup += count
    print(f"  {label:<25s}: {count:>4d}条")

print(f"  {'합계':<25s}: {total_sup:>4d}条")

# 2. Check final training data
print()
print("=" * 60)
print("最终训练数据 train_data_all_final.json 统计")
print("=" * 60)

data = json.load(open(BASE / "train_data_all_final.json", "r", encoding="utf-8"))
print(f"  总条数: {len(data)}")

# Sample entries from each weak category
print()
print("=" * 60)
print("各弱项补充后 实际覆盖数量（在最终8072条中）")
print("=" * 60)

scenarios = {
    "치과 (牙科)": ["치과", "이빨", "잇몸", "충치", "때우다", "스케일링", "임플란트", "틀니", "어금니", "앞니"],
    "정형외과 (骨科)": ["뼈", "골절", "깁스", "목발", "부러지다", "디스크", "허리", "부목"],
    "장애인편의 (残疾人)": ["휠체어", "보조기", "보청기", "장애", "수어", "통역", "필담", "안내견"],
    "응급실 (急诊)": ["응급", "급하다", "구급", "119", "쓰러지다"],
    "약국 (药房)": ["약국", "처방전", "약사"],
}

for label, kws in scenarios.items():
    count = sum(1 for e in data if any(k in " ".join(e["words"]) + " " + e["reference"] for k in kws))
    print(f"  {label:<25s}: {count:>4d}条")

# 3. Show random samples from supplement
print()
print("=" * 60)
print("补充数据 随机样例 (每类3条)")
print("=" * 60)

import random
random.seed(42)
sup_data = json.load(open(BASE / "train_data_supplement.json", "r", encoding="utf-8"))

# Group by keywords
for label, prefix in [("치과", "sup_치과"), ("정형외과", "sup_정형"), ("장애인", "sup_장애"), ("응급실", "sup_응급"), ("약국", "sup_약국")]:
    entries = []
    for f in sorted(batch_dir.glob(f"{prefix}*.json")):
        if f.name.startswith("_"): continue
        try: entries.extend(json.loads(f.read_text(encoding="utf-8")))
        except: pass
    if entries:
        samples = random.sample(entries, min(3, len(entries)))
        print(f"\n  [{label}]")
        for s in samples:
            print(f"    words: {s['words']}")
            print(f"    ref:   {s['reference']}")
