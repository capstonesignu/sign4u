# AI-sLLM: 한국수어 → 한국어 문장 변환 (Korean Sign Language Gloss-to-Text)

수어 인식(SLR)이 내놓은 단어 시퀀스를, 파인튜닝한 LLM과 7단계 제어 추론 파이프라인을 거쳐 자연스러운 한국어 문장으로 변환한다.

```
수어 영상 → 키포인트 → SLR → 단어 시퀀스 → sLLM → 한국어 문장
```

## 어체: 합쇼체 (존댓말)

모델 출력은 모두 **합쇼체(`~습니다 / ~ㅂ니다`)** 로 통일된다.

### 최신 모델 — v5-cons (2026-06)
학습 데이터에 섞여 있던 ‘입력에 없는 내용을 덧붙이던’ 문장 **587개를 정제**한 뒤 다시 학습시킨 버전이다. 이전 v4에서 문제였던 **환각(입력에 없는 단어 생성)이 줄었고**, 합쇼체율 100% · 영어 혼입 0%를 보인다.

- **어댑터**: `exaone-finetuned-v5-cons` (EXAONE-3.5-2.4B + QLoRA 4bit, r16, 4 epoch) — 용량 문제로 git이 아닌 **Google Drive**로 공유
- **학습 데이터**: `train_data_all_final_chat_v5_conservative.jsonl` (7,485문장, 합쇼체 · 정제본)
- **평가** (`test_data_clean.json`, 96문장): 수용 가능률 약 **69%**(LLM 심사 기준) · 합쇼체율 **100%** · keyword recall 92.6%
- **추론 속도(4bit)**: 단일 greedy 약 **0.36초/문장** (1초 미만 ✓)

### 참고 — 이전 버전 / 폐기된 실험
- **v4** (2026-06-03): 합쇼체율 97.9% · BLEU 0.285. v5는 이 v4를 데이터 정제로 개선한 버전이다.
- **모델 크기**: 7.8B도 테스트했으나 환각이 오히려 많아 **2.4B 4bit**를 유지한다(데이터로 뒷받침된 결정).
- **후보(top-k) 학습 (v6-cand)**: SLR이 위치별 후보를 줄 때를 가정한 파인튜닝도 실험했으나, 단일 단어 입력 품질이 떨어지는 trade-off가 확인되어 채택하지 않음. **최종 배포 인터페이스는 top-1 단어 시퀀스 + `v5-cons` 단일 모델**(상세: `docs/report/SLR-sLLM接口约定.md`).

## 빠른 시작 (Quick Start)

```bash
# 의존성 설치
pip install -r requirements.txt

# 단일 입력 테스트 (룰 기반, GPU 불필요)
python sllm_module.py --words 나 학교 오늘 가다

# JSON 입력
python sllm_module.py --input-json '{"request_id":"001","words":["나","학교","오늘","가다"]}'

# 전체 평가 (GPU 필요)
python evaluate.py --data test_data_clean.json --backends rule finetuned
```

> ⚠️ Windows에서는 인코딩 문제로 `python -X utf8 ...` 로 실행할 것을 권장한다.

## 통합 사용 (Integration)

```python
from sllm_module import SllmModule

sllm = SllmModule(backend="finetuned")  # 또는 "rule" (GPU 불필요)
result = sllm.normalize(["나", "학교", "오늘", "가다"])
print(result)  # "나는 오늘 학교에 갑니다."
```

**입출력 형식:**

```json
// 입력
{"request_id": "001", "words": ["나", "학교", "오늘", "가다"]}

// 출력
{"request_id": "001", "normalized_text": "나는 오늘 학교에 갑니다.", "status": "success"}
```

## 7단계 파이프라인 (7-Layer Pipeline)

파인튜닝 백엔드는 각 입력을 다음 7단계로 처리한다.

| 단계 | 파일 | 역할 |
|------|------|------|
| 1. 원시 생성 | `sllm_module.py` | LLM으로 여러 후보 문장 생성 |
| 2. 제약 채점 | `constraint_scorer.py` | 키워드/부정/길이 기준으로 후보 채점 |
| 3. 후보 선택 | `candidate_selector.py` | 가중 점수로 최적 후보 선택 |
| 4. 형식 정리 | `format_cleanup.py` | 프롬프트 잔여물 제거, 문장부호 정리 |
| 5. 의미 정리 | `semantic_cleanup.py` | 시제 교정 · 영어→한국어 · 문법 교정 · 환각 제거 (롤백 보호 포함) |
| 6. 최종 가드레일 | `guardrail.py` | 형식·의미 안전성 점검, 폴백 전환 |
| 7. 트레이스 | `inference_trace.py` | 디버깅용 전체 파이프라인 로깅 |

## 평가 · 데이터 정제 도구 (v5에서 추가)

| 파일 | 역할 |
|------|------|
| `scripts/reaudit.py` | GPU 없이 평가 결과 재감사 — 재현 가능한 kiwi 형태소 기반 환각 지표 + 띄어쓰기 점검 |
| `scripts/clean_traindata.py` | 환각을 유발하는 학습 문장 분류·정제 (actor / extra / forced_predicate) |
| `scripts/make_candidate_traindata_v5.py` | 정제된 데이터로 후보(top-k) 형식 학습 데이터 생성 |
| `scripts/check_honorific.py` | 합쇼체율 측정 |

## 디렉터리 구조 (File Structure)

```
AI-sLLM/
├── sllm_module.py              # 메인 모듈 (rule / exaone / finetuned 백엔드)
├── constraint_scorer.py        # 2단계: 제약 채점
├── candidate_selector.py       # 3단계: 후보 선택
├── format_cleanup.py           # 4단계: 형식 정리
├── semantic_cleanup.py         # 5단계: 의미 정리 + 롤백
├── guardrail.py                # 6단계: 최종 가드레일
├── inference_trace.py          # 7단계: 트레이스 로깅
├── failure_taxonomy.py         # 14가지 실패 유형 정의
├── prompt_templates.py         # 프롬프트 템플릿 (basic / few-shot / candidate)
├── medical_vocab.py            # 의료 어휘 사전
├── evaluate.py                 # 자동 평가 (BLEU, chrF++ 등)
├── finetune_v3.py              # QLoRA 파인튜닝 스크립트
├── requirements.txt            # Python 의존성
│
├── config/
│   ├── cleanup_config.py       # 정리 단계 토글 스위치
│   ├── generation_config.py    # LLM 생성 파라미터
│   └── scoring_config.py       # 제약 채점 가중치
│
├── scripts/
│   ├── reaudit.py              # (v5) GPU 없는 평가 재감사 · 환각 지표
│   ├── clean_traindata.py      # (v5) 학습 데이터 환각 정제
│   ├── make_candidate_traindata_v5.py  # (v5) 후보 형식 학습 데이터 생성
│   ├── check_honorific.py      # (v5) 합쇼체율 측정
│   ├── run_human_eval.py       # 사람 평가 실행 (96문항)
│   ├── error_analysis.py       # 오류 원인별 분류
│   └── pre_freeze_audit.py     # 코드 프리즈 전 감사 (7단계 검증)
│
├── evaluation/                 # 평가 결과물 (human eval, 오류 분석 등)
├── tests/                      # 단위 테스트 (test_constraints.py 등)
├── deprecated/                 # 구버전 파일 (7단계 파이프라인으로 대체됨)
│
├── test_data_clean.json                       # 테스트 데이터 (96문항, 데이터 누수 0%)
├── train_data_all_final.json                  # 학습 데이터 원본 (8,072문항)
├── train_data_all_final_chat.jsonl            # 학습 데이터 (chat 형식)
└── train_data_all_final_chat_v5_conservative.jsonl  # (v5) 정제 학습 데이터 (7,485문항)
```

## 백엔드 (Backends)

| 백엔드 | 설명 | GPU |
|--------|------|-----|
| `rule` | 룰 기반 베이스라인 (조사 삽입 + 활용) | 불필요 |
| `exaone` | EXAONE-3.5-2.4B-Instruct (기본 모델) | 필요 |
| `finetuned` | EXAONE + QLoRA 파인튜닝 + 7단계 파이프라인 | 필요 |

## 파인튜닝 (Fine-tuning)

```bash
# 8GB 이상 VRAM GPU 필요. Windows에서는 -X utf8 권장.
python -X utf8 finetune_v3.py \
    --data-path train_data_all_final_chat_v5_conservative.jsonl \
    --output-dir exaone-finetuned-v5-cons
```

학습된 어댑터는 용량 때문에 gitignore되며, **Google Drive**로 공유한다.

## 테스트 (Tests)

```bash
python -m pytest tests/ -v
```

## 버전 (Version)

- **v5-cons** (최신) — 학습 데이터 정제로 환각 감소, 합쇼체율 100%, 영어 혼입 0%
- 모델: EXAONE-3.5-2.4B-Instruct + QLoRA (4-bit NF4)
- 학습 7,485문항 / 테스트 96문항 (데이터 누수 0%)
