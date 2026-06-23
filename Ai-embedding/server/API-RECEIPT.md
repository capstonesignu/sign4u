# KSL Production API 명세서

## `GET /health`

서버 상태 확인.

**Response** `200 OK`

```json
{
  "status": "ok",
  "num_vectors": 63,
  "num_words": 21,
  "encoder": "capstone/encoder/encoder_varlen_best.pt",
  "db": "capstone/db/recorded_only.faiss",
  "repo": "capstoneSign/ksl-models"
}
```

---

## `POST /predict`

단어 세그먼트 목록을 받아 SSE로 인식 결과 스트리밍.

**Content-Type** `application/json`  
**Response** `text/event-stream`

### Request Body

```json
{
  "words": [
    {
      "pose":       [[...], ...],
      "left_hand":  [[...], ...],
      "right_hand": [[...], ...],
      "face":       [[...], ...]
    }
  ]
}
```

| 필드 | 타입 | 형상 | 설명 |
|------|------|------|------|
| `words` | `WordSegment[]` | min 1개 | 시간 순 단어 목록 |
| `pose` | `float[][]` | `(T, 27)` | 9 landmarks × 3 axes |
| `left_hand` | `float[][]` | `(T, 63)` | 21 landmarks × 3 axes |
| `right_hand` | `float[][]` | `(T, 63)` | 21 landmarks × 3 axes |
| `face` | `float[][]` | `(T, 57)` | 19 landmarks × 3 axes |

T는 단어마다 달라도 됨 (서버 내부에서 64프레임으로 보간).

---

### SSE 이벤트 형식

모든 이벤트의 공통 구조:

```
data: {"status": "<status>", "content": {...}}\n\n
```

### SSE 이벤트 목록

| status | 시점 | content 필드 | 설명 |
|--------|------|-------------|------|
| `received` | 요청 수신 직후 | `n_words` | 총 단어 수 |
| `rag-retrieve` | 단어 1개 처리 완료마다 | `idx`, `total` | FAISS 검색 완료 알림 |
| `generate-sentence` | 전체 RAG 완료, sLLM 시작 전 | (없음) | 문장 생성 시작 알림 |
| `sentence-chunk` | 문장 글자 1개마다 | `idx`, `chunk` | sLLM 출력 스트리밍 |
| `done` | 전체 완료 | (없음) | 스트림 종료 |
| `error` | 단어 단위 오류 | `idx`, `message` | 해당 단어만 스킵, 스트림 계속 |

### content 상세

**`received`**
```json
{ "n_words": 3 }
```

**`rag-retrieve`**
```json
{ "idx": 0, "total": 3 }
```

**`generate-sentence`**
```json
{}
```

**`sentence-chunk`**
```json
{ "idx": 0, "chunk": "나" }
```

**`done`**
```json
{}
```

**`error`**
```json
{ "idx": 1, "message": "프레임 부족 (< 2)" }
```

---

### 이벤트 순서 예시 (3단어 입력)

```
data: {"status": "received",         "content": {"n_words": 3}}

data: {"status": "rag-retrieve",     "content": {"idx": 0, "total": 3}}
data: {"status": "rag-retrieve",     "content": {"idx": 1, "total": 3}}
data: {"status": "rag-retrieve",     "content": {"idx": 2, "total": 3}}

data: {"status": "generate-sentence","content": {}}

data: {"status": "sentence-chunk",   "content": {"idx": 0, "chunk": "나"}}
data: {"status": "sentence-chunk",   "content": {"idx": 1, "chunk": "는"}}
data: {"status": "sentence-chunk",   "content": {"idx": 2, "chunk": " "}}
data: {"status": "sentence-chunk",   "content": {"idx": 3, "chunk": "병"}}
...

data: {"status": "done",             "content": {}}
```
