# Ai-embedding

수어(KSL) → 한국어 번역 파이프라인의 AI 모듈. 웹캠 키포인트를 임베딩하여 FAISS로 단어를
검색(RAG)하고, sLLM이 자연스러운 한국어 문장으로 변환한다. 학습 시에만 GPU를 사용하고
서비스 추론은 CPU만으로 동작한다.

```
키포인트(MediaPipe) → Cross-Stream 인코더(임베딩) → FAISS 검색 → sLLM 문장 생성
```

## 구조

```
Ai-embedding/
├── server/      [배포] AI 추론 서버 (Railway). 인식 + sLLM 오케스트레이션, SSE 스트리밍
├── sllm/        [배포] sLLM 모델 서버 (로컬 + ngrok). EXAONE 3.5 2.4B GGUF, llama.cpp
└── training/    [재생성] 인코더 학습 · FAISS DB 빌드 · 진단/평가 도구
```

모델 weight(인코더 .pt, FAISS .faiss, EXAONE gguf 등)는 용량 문제로 git에 포함하지 않는다.
인코더·DB는 HuggingFace에 올려 `server/`가 기동 시 다운로드하고, sLLM weight는 별도 배포한다.

## 배포

### 1) 인식 서버 — `server/` (Railway, Dockerfile)
기동 시 HuggingFace에서 인코더·FAISS DB를 받고, 문장 생성은 `SLLM_ENDPOINT`(아래 sllm 서버)로
위임한다. 필요한 환경변수(기본값 없음 — 미설정 시 기동 실패):

```
HF_TOKEN, HF_REPO_ID, ENCODER_FILENAME, FAISS_FILENAME, FAISS_LABELS_FILENAME,
FEATURE_PRESET=B, TOP_K, SLLM_ENDPOINT=<sLLM ngrok URL>, HOST, PORT
```
`.env.example` 참고. HF 경로 규약: `capstone/encoder/<ENCODER>`, `capstone/db/<FAISS>`.

### 2) sLLM 서버 — `sllm/` (로컬 + ngrok)
EXAONE 3.5 2.4B(LoRA 파인튜닝)를 GGUF Q4_K_M로 llama.cpp 추론. 로컬에서 실행 후 ngrok으로
노출하고, 그 URL을 server 의 `SLLM_ENDPOINT`에 설정한다. 모델 weight(gguf)는 git 미포함 —
`make_gguf.py`로 생성하거나 별도 배포본을 받는다.

### 3) 모델 아티팩트 → HuggingFace
```
python training/upload_to_hf.py --repo <org>/<repo>
```
cross-stream 인코더 + flip-free FAISS DB 를 업로드한다.

## 인코더 재생성 (training/)

```
# 의존성
pip install -r training/requirements.txt

# 학습 (AI Hub 키포인트 필요 — dataset 은 git 미포함)
python training/train_cs_collapse.py --output result_cs_collapse

# 평가 (recorded-video LOO)
python training/eval_tta_loo.py --model result_cs_collapse/encoder_cross_stream_best.pt

# FAISS DB 빌드 (flip-free)
python training/build_jbedu_db.py --out-dir db_staging

# 임베딩 공간 진단 / 인코더 비교 / 스윕 플롯 / 녹화 QC
python training/diagnose_embedding.py
python training/compare_encoders.py
```

모델: 4-스트림 Cross-Attention Transformer(d_model=256, 약 4.8M params), 입력 Preset B
136차원(64프레임), 출력 256차원 L2 정규화. NT-Xent + VICReg(variance/covariance)로 학습하여
dimensional collapse를 억제한다.

## 미포함 (의도적)
- `dataset/`(학습·평가 데이터), `result_*/`(체크포인트), 모든 모델 weight → 용량/HF 분리
- 로컬 테스트용 데모(`demo/`, `demo-desktop/`)
- `server/services/segmenter.py`의 BiLSTM 세그멘터는 현재 `/predict`(단어 단위)에서 미사용이며,
  의존하는 `seg_model` 가중치는 번들하지 않는다.
