"""Prompt engineering benchmark — v5-cons 어댑터, 합쇼체 기준.

모델을 한 번 로드하고 여러 프롬프트 전략을 테스트한다.
"""
import os, re, time, statistics, torch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE    = str(Path(__file__).parent / "base_model")
ADAPTER = str(
    Path(__file__).parent.parent.parent / "AI-sLLM" / "exaone-finetuned-v5-cons"
)
DEVICE  = "cpu"

_NEGATION_WORDS = {"없다", "없음", "안", "못", "못하다", "불가능", "불가", "필요없다"}


def _patch_emb(model):
    t = getattr(model, "transformer", None)
    if t:
        t.__class__.get_input_embeddings = lambda self: self.wte
        t.__class__.set_input_embeddings = lambda self, v: setattr(self, "wte", v)
    try:
        model.get_output_embeddings()
    except Exception:
        model.__class__.get_output_embeddings = lambda self: getattr(self, "lm_head", None)


print("[bench] loading model (one-time)…")
t0 = time.time()

if Path(BASE).exists():
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True
    )
else:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        quantization_config=bnb, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

_patch_emb(base)
model     = PeftModel.from_pretrained(base, ADAPTER)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
model.eval()
print(f"[bench] ready in {time.time()-t0:.0f}s\n")

# ── 실제 테스트 후보 ──────────────────────────────────────────────────────────
CANDIDATES = [
    [{"word": "아프다", "score": 0.95}, {"word": "통증",   "score": 0.82}],
    [{"word": "머리",   "score": 0.92}, {"word": "눈",     "score": 0.78}],
    [{"word": "감기",   "score": 0.90}, {"word": "몸살",   "score": 0.75}],
]


def fmt_pos(cands):
    lines = []
    for i, pos in enumerate(cands, 1):
        if not pos:
            continue
        parts = " | ".join(f"{c['word']}({c['score']:.2f})" for c in pos)
        lines.append(f"위치 {i}: {parts}")
    return "\n".join(lines)


def infer(system_msg, user_text, max_new_tokens):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_text},
    ]
    fmt    = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(fmt, return_tensors="pt")
    input_ids = inputs.input_ids.to(DEVICE)
    attn      = inputs.attention_mask.to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    new = out[0][input_ids.shape[-1]:]
    return tokenizer.decode(new, skip_special_tokens=True).strip(), inputs.input_ids.shape[-1]


def parse(raw):
    sents = []
    for line in raw.splitlines():
        m = re.match(r"^\d+\.\s*(.+)$", line.strip())
        if m:
            sents.append(m.group(1).strip())
    if not sents:
        sents = [s.strip() for s in raw.splitlines() if s.strip()]
    return sents[:3]


def run(name, system_msg, max_new_tokens, n=5):
    user = f"입력:\n{fmt_pos(CANDIDATES)}\n\n출력:"
    times, ilen = [], 0
    raw = ""
    for _ in range(n):
        t0 = time.time()
        raw, ilen = infer(system_msg, user, max_new_tokens)
        times.append(time.time() - t0)
    sents = parse(raw)
    avg, std = statistics.mean(times), statistics.stdev(times)
    flag = " ✓" if avg <= 5.0 else ""
    print(f"\n{'='*60}")
    print(f"[{name}]  max_new_tokens={max_new_tokens}  input_tokens={ilen}{flag}")
    print(f"  평균={avg:.2f}s  편차={std:.2f}s  min={min(times):.2f}s  max={max(times):.2f}s")
    print(f"  출력: {sents}")
    return avg, std, sents


results = []

# ── Few-shot 예시 (합쇼체) ───────────────────────────────────────────────────
EX1 = [
    [{"word": "배",    "score": 0.95}, {"word": "복통",  "score": 0.82}],
    [{"word": "아프다","score": 0.90}, {"word": "통증",  "score": 0.85}],
    [{"word": "병원",  "score": 0.98}, {"word": "의원",  "score": 0.80}],
    [{"word": "가다",  "score": 0.92}, {"word": "방문",  "score": 0.75}],
]
EX1_OUT = ["배가 아파서 병원에 갑니다.", "복통이 있어서 병원에 갑니다.", "배가 아파서 의원에 갑니다."]

EX2 = [
    [{"word": "처방전","score": 0.96}, {"word": "처방",  "score": 0.80}],
    [{"word": "받다",  "score": 0.93}, {"word": "수령",  "score": 0.70}],
    [{"word": "약국",  "score": 0.98}, {"word": "병원",  "score": 0.65}],
    [{"word": "가다",  "score": 0.94}, {"word": "방문",  "score": 0.78}],
]
EX2_OUT = ["처방전을 받고 약국에 갑니다.", "처방을 받고 약국에 갑니다.", "처방전을 받고 약국에 방문합니다."]


def fewshot_block(examples):
    parts = []
    for cands, sents in examples:
        pos = fmt_pos(cands)
        num = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sents))
        parts.append(f"입력:\n{pos}\n\n{num}")
    return "\n\n---\n\n".join(parts)


BASE_RULES = (
    "너는 한국수어 번역 시스템의 후처리 sLLM이다.\n"
    "각 위치마다 인식 후보 단어들과 유사도 점수가 주어진다.\n"
    "가장 자연스러운 한국어 문장을 3개 생성하라.\n\n"
    "### 필수 규칙 ###\n"
    "1. 반드시 합쇼체(-습니다, -ㅂ니다)만 사용한다.\n"
    "2. 과거 표현이 없으면 현재형으로 쓴다.\n"
    "3. 각 위치의 후보 단어 중 하나만 선택하고, 후보에 없는 단어는 절대 추가하지 않는다.\n"
    "4. 반드시 '1. 문장\\n2. 문장\\n3. 문장' 형식으로만 출력한다.\n"
    "5. 각 문장은 서로 다른 후보 단어 조합을 사용하라.\n"
    "6. 안/못/없다 등 부정어가 후보에 있으면 반드시 부정 표현을 유지하라.\n"
)

SHORT_RULES = (
    "너는 한국수어 번역 시스템의 후처리 sLLM이다.\n"
    "각 위치의 후보 단어로 자연스러운 한국어 문장 3개를 생성하라.\n\n"
    "규칙: 합쇼체(-습니다)만 사용. 과거표현 없으면 현재형. 후보에 없는 단어 추가 금지.\n"
    "출력 형식: '1. 문장\\n2. 문장\\n3. 문장' 형식만 사용. 각 문장은 다른 단어 조합.\n"
)

# ── v5-cons 단일 문장 프롬프트 (top-K 입력 → 문장 1개) ───────────────────────
SINGLE_RULES = (
    "너는 한국수어 번역 시스템의 후처리 sLLM이다.\n"
    "각 위치마다 인식 후보 단어들과 유사도 점수가 주어진다.\n"
    "점수가 높은 단어를 우선 사용해 자연스러운 한국어 문장 하나를 만들어라.\n\n"
    "규칙:\n"
    "1. 반드시 합쇼체(-습니다, -ㅂ니다)만 사용한다.\n"
    "2. 각 위치의 후보 단어 중 하나만 선택하고, 후보에 없는 단어는 절대 추가하지 않는다.\n"
    "3. 설명 없이 문장 하나만 출력한다.\n"
    "4. 반드시 한국어만 출력한다."
)

# A: 3-sentence, 3 few-shot, max=240
sys_A = BASE_RULES + "\n### 예시 ###\n\n" + fewshot_block([(EX1, EX1_OUT), (EX2, EX2_OUT), (EX1, EX1_OUT)]) + "\n\n"
r = run("A-3shot-240", sys_A, 240); results.append(("A-3shot-240", *r[:2]))

# B: 3-sentence, 1 few-shot, max=120
sys_B = BASE_RULES + "\n### 예시 ###\n\n" + fewshot_block([(EX1, EX1_OUT)]) + "\n\n"
r = run("B-1shot-120", sys_B, 120); results.append(("B-1shot-120", *r[:2]))

# C: 3-sentence, 1 few-shot, max=90
r = run("C-1shot-90", sys_B, 90); results.append(("C-1shot-90", *r[:2]))

# D: 3-sentence, 0 few-shot (SHORT rules), max=90
r = run("D-0shot-short-90", SHORT_RULES, 90); results.append(("D-0shot-short-90", *r[:2]))

# E: 3-sentence, 0 few-shot (BASE rules), max=90
r = run("E-0shot-base-90", BASE_RULES, 90); results.append(("E-0shot-base-90", *r[:2]))

# F: 3-sentence, 1 few-shot + SHORT rules, max=90
sys_F = SHORT_RULES + "\n예시:\n" + fewshot_block([(EX1, EX1_OUT)]) + "\n\n"
r = run("F-1shot-short-90", sys_F, 90); results.append(("F-1shot-short-90", *r[:2]))

# G: 3-sentence, 1 few-shot + SHORT rules, max=80
r = run("G-1shot-short-80", sys_F, 80); results.append(("G-1shot-short-80", *r[:2]))

# H: 단일 문장 (v5-cons 학습 방식), 1 few-shot, max=64
EX1_SINGLE = [
    (
        [
            [{"word": "배",   "score": 0.95}, {"word": "복통", "score": 0.82}],
            [{"word": "아프다","score": 0.90}, {"word": "통증", "score": 0.85}],
            [{"word": "병원", "score": 0.98}, {"word": "의원", "score": 0.80}],
            [{"word": "가다", "score": 0.92}, {"word": "방문", "score": 0.75}],
        ],
        "배가 아파서 병원에 갑니다.",
    ),
]

def fewshot_single(examples):
    parts = []
    for cands, sent in examples:
        pos = fmt_pos(cands)
        parts.append(f"입력:\n{pos}\n\n출력: {sent}")
    return "\n\n---\n\n".join(parts)

sys_H = SINGLE_RULES + "\n### 예시 ###\n\n" + fewshot_single(EX1_SINGLE) + "\n\n### 변환 ###\n"
r = run("H-single-1shot-64", sys_H, 64); results.append(("H-single-1shot-64", *r[:2]))

print("\n\n" + "="*60)
print(f"{'Strategy':<24} {'avg':>6} {'std':>6}")
print("-"*40)
for name, avg, std in results:
    flag = " ✓" if avg <= 5.0 else ""
    print(f"{name:<24} {avg:>6.2f}s {std:>6.2f}s{flag}")
