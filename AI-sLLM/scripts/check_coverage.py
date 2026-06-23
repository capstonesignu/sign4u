"""Check medical scenario coverage in merged training data."""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')
data = json.load(open('train_data_final_merged.json', 'r', encoding='utf-8'))

scenarios = {
    '접수/등록 (Registration)': ['접수', '등록', '번호표', '대기', '창구'],
    '보험/수납 (Insurance/Pay)': ['보험', '수납', '영수증', '비용'],
    '응급실 (Emergency)': ['응급', '급하다', '구급'],
    '수어소통 (SignLang Comm)': ['수어', '통역', '수화', '필담', '의사소통'],
    '재활치료 (Rehab)': ['재활', '물리치료', '스트레칭', '근력'],
    '치과 (Dental)': ['치과', '이빨', '잇몸', '충치'],
    '안과 (Eye)': ['안과', '시력', '안경', '렌즈', '안약'],
    '정형외과 (Orthopedic)': ['뼈', '골절', '깁스', '목발', '부러지다'],
    '산부인과 (OB/GYN)': ['임신', '아기', '출산', '태아'],
    '정신건강 (Mental)': ['우울', '불안', '스트레스', '상담', '심리'],
    '장애인편의 (Disability)': ['휠체어', '보조기', '보청기', '장애'],
    '약국 (Pharmacy)': ['약국', '처방전', '약사'],
    '입원/퇴원 (Hospital Stay)': ['입원', '퇴원', '병실', '간호사'],
    '수술 (Surgery)': ['수술', '마취', '회복'],
    '검사 (Tests)': ['검사', '피뽑다', '소변', '엑스레이', '초음파'],
    '약 복용 (Medication)': ['약', '먹다', '알약', '캡슐', '가루약'],
    '통증표현 (Pain)': ['아프다', '쑤시다', '찌르다', '욱신'],
    '증상묘사 (Symptoms)': ['열', '기침', '콧물', '구토', '설사', '어지럽다'],
    '진료대화 (Consultation)': ['의사', '진료', '진단', '치료', '설명'],
    '퇴원교육 (Discharge Ed)': ['퇴원', '주의', '조심', '운동', '식단'],
}

print(f"Total training entries: {len(data)}\n")
print(f"{'Scenario':<30s} {'Count':>5s}  {'%':>5s}")
print("-" * 45)

weak = []
for name, kws in scenarios.items():
    count = sum(1 for e in data
                if any(k in ' '.join(e['words']) + ' ' + e['reference'] for k in kws))
    pct = count / len(data) * 100
    marker = " ⚠️ WEAK" if count < 30 else ""
    print(f"{name:<30s} {count:>5d}  {pct:>4.1f}%{marker}")
    if count < 30:
        weak.append((name, count, kws))

print(f"\n{'=' * 45}")
print(f"Weak areas (< 30 entries): {len(weak)}")
for name, count, kws in weak:
    print(f"  {name}: only {count} entries - keywords: {kws}")
