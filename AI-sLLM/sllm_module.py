"""AI-sLLM Words-to-Text module.

7-layer controlled inference pipeline:
  Layer 1: Raw Generation      — prompt + model inference (1 greedy + N sampling)
  Layer 2: Constraint Scoring  — score each candidate on keyword/negation/hallucination/length
  Layer 3: Candidate Selection — weighted score ranking, pick best + fallback
  Layer 4: Format Cleanup      — pure formatting (prefix, spacing, clipping, polite→plain)
  Layer 5: Semantic Cleanup    — semantic repair (English→Korean, tense, grammar, hallucination)
  Layer 6: Final Guardrail     — format + semantic safety check, fallback if needed
  Trace:   Inference tracing   — full pipeline record for debugging

Backends:
  - RuleBasedGenerator   : conservative lookup table (no GPU needed)
  - ExaoneGenerator      : EXAONE-3.5-2.4B-Instruct via Hugging Face (GPU)
  - FinetunedGenerator   : EXAONE + QLoRA fine-tuned adapter (GPU)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from config import GenerationConfig, ScoringConfig, CleanupConfig
from prompt_templates import build_basic_prompt, build_fewshot_prompt
from constraint_scorer import ConstraintScorer
from candidate_selector import CandidateSelector
from format_cleanup import FormatCleaner
from semantic_cleanup import SemanticCleaner
from guardrail import FinalGuardrail
from inference_trace import InferenceTrace, TraceRecorder

# Old postprocess module moved to deprecated/ — import only if available
try:
    from deprecated.postprocess import clean_model_output, validate_sentence  # noqa: F401
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Backend 1: Rule-based (offline fallback)
# ---------------------------------------------------------------------------

class RuleBasedGenerator:
    """Small deterministic fallback for offline integration tests."""

    EXACT = {
        ("나", "학교", "오늘", "가다"): "나는 오늘 학교에 간다.",
        ("내일", "비", "오다", "우산", "필요"): "내일 비가 오니 우산이 필요하다.",
        ("친구", "만나다", "기쁘다"): "친구를 만나서 기쁘다.",
        ("엄마", "병원", "가다"): "엄마가 병원에 간다.",
        ("아빠", "회사", "가다"): "아빠가 회사에 간다.",
        ("나", "밥", "먹다"): "나는 밥을 먹는다.",
        ("너", "어디", "가다"): "너는 어디에 가니?",
        ("오늘", "날씨", "좋다"): "오늘 날씨가 좋다.",
        ("수업", "끝나다", "집", "가다"): "수업이 끝나고 집에 간다.",
        ("도움", "필요"): "도움이 필요하다.",
        ("나", "머리", "아프다", "약", "먹다"): "나는 머리가 아파서 약을 먹는다.",
        ("배", "아프다", "병원", "가다"): "배가 아파서 병원에 간다.",
        ("감기", "걸리다", "쉬다"): "감기에 걸려서 쉰다.",
        ("물", "마시다", "목", "마르다"): "목이 말라서 물을 마신다.",
        ("버스", "타다", "학교", "가다"): "버스를 타고 학교에 간다.",
        ("나", "책", "읽다", "좋아하다"): "나는 책 읽는 것을 좋아한다.",
        ("오늘", "시험", "있다", "긴장"): "오늘 시험이 있어서 긴장된다.",
        ("어제", "영화", "보다", "재미있다"): "어제 영화를 봤는데 재미있었다.",
        ("나", "한국어", "배우다"): "나는 한국어를 배운다.",
        ("지하철", "사람", "많다"): "지하철에 사람이 많다.",
        ("나", "피곤하다", "자다"): "나는 피곤해서 잔다.",
        ("선생님", "설명", "이해", "못하다"): "선생님의 설명을 이해하지 못한다.",
        ("카페", "커피", "마시다"): "카페에서 커피를 마신다.",
        ("동생", "울다", "슬프다"): "동생이 슬퍼서 운다.",
        ("주말", "가족", "여행", "가다"): "주말에 가족과 여행을 간다.",
        ("진료", "예약", "하다"): "진료를 예약한다.",
        ("접수", "하다", "어디", "가다"): "접수를 하려면 어디에 가야 하는지 모른다.",
        ("초진", "접수", "서류", "작성"): "초진 접수를 위해 서류를 작성한다.",
        ("머리", "아프다", "열", "나다"): "머리가 아프고 열이 난다.",
        ("목", "아프다", "기침", "나다"): "목이 아프고 기침이 난다.",
        ("허리", "아프다", "걷다", "힘들다"): "허리가 아파서 걷기 힘들다.",
        ("소화", "안", "되다", "속", "불편하다"): "소화가 안 되고 속이 불편하다.",
        ("눈", "아프다", "충혈"): "눈이 아프고 충혈되었다.",
        ("어지럽다", "쓰러지다", "넘어지다"): "어지러워서 쓰러지고 넘어졌다.",
        ("혈압", "측정", "하다"): "혈압을 측정한다.",
        ("주사", "맞다", "아프다", "참다"): "주사를 맞고 아프지만 참는다.",
        ("엑스레이", "찍다", "결과", "기다리다"): "엑스레이를 찍고 결과를 기다린다.",
        ("피", "검사", "하다"): "피 검사를 한다.",
        ("CT", "찍다", "예약"): "CT를 찍기 위해 예약한다.",
        ("약", "먹다", "하루", "세", "번"): "약을 하루에 세 번 먹는다.",
        ("처방전", "받다", "약국", "가다"): "처방전을 받고 약국에 간다.",
        ("알레르기", "있다", "약", "못", "먹다"): "알레르기가 있어서 약을 못 먹는다.",
        ("진통제", "먹다", "통증", "줄다"): "진통제를 먹고 통증이 줄었다.",
        ("수술", "후", "회복", "중"): "수술 후 회복 중이다.",
        ("퇴원", "언제", "가능"): "퇴원은 언제 가능한지 묻는다.",
        ("입원", "며칠", "필요"): "입원이 며칠 필요하다.",
        ("수술", "동의서", "서명", "하다"): "수술 동의서에 서명한다.",
    }

    SUBJECTS = {"나": "나는", "너": "너는", "우리": "우리는", "엄마": "엄마가", "아빠": "아빠가", "친구": "친구가"}
    PLACES   = {"학교": "학교에", "병원": "병원에", "회사": "회사에", "집": "집에"}
    OBJECTS  = {"밥": "밥을", "물": "물을", "책": "책을", "도움": "도움이"}
    VERBS    = {
        "가다": "간다", "오다": "온다", "먹다": "먹는다", "마시다": "마신다",
        "보다": "본다", "필요": "필요하다", "좋다": "좋다",
        "끝나다": "끝난다", "만나다": "만난다", "기쁘다": "기쁘다",
    }

    def generate(self, words: list[str], prompt: str = "") -> str:
        key = tuple(words)
        if key in self.EXACT:
            return self.EXACT[key]

        normalized: list[str] = []
        for word in words:
            if word in self.SUBJECTS:
                normalized.append(self.SUBJECTS[word])
            elif word in self.PLACES:
                normalized.append(self.PLACES[word])
            elif word in self.OBJECTS:
                normalized.append(self.OBJECTS[word])
            elif word in self.VERBS:
                normalized.append(self.VERBS[word])
            else:
                normalized.append(word)

        sentence = " ".join(normalized).strip()
        if not sentence:
            return ""
        if sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    def generate_with_sampling(self, words: list[str], prompt: str = "") -> str:
        """Rule-based has no sampling — returns same as greedy."""
        return self.generate(words, prompt)


# ---------------------------------------------------------------------------
# Backend 2: EXAONE 3.5 2.4B Instruct (real sLLM)
# ---------------------------------------------------------------------------

class ExaoneGenerator:
    """Generate Korean sentences using EXAONE-3.5-2.4B-Instruct."""

    MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    def __init__(self, gen_config: GenerationConfig | None = None, model_id: str = ""):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.gen_config = gen_config or GenerationConfig()
        self._model_id = model_id or self.MODEL_ID

        print(f"[sLLM] Loading model: {self._model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.model.generation_config.max_new_tokens = self.gen_config.max_new_tokens_base
        self.model.generation_config.do_sample = False
        self.model.generation_config.repetition_penalty = self.gen_config.repetition_penalty
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        print(f"[sLLM] Model loaded successfully on {self.model.device}")

    def _build_messages(self, prompt: str) -> list[dict]:
        return [
            {"role": "system", "content": (
                "너는 한국수어 번역 시스템의 후처리 sLLM이다. "
                "단어 시퀀스를 한국어 문장 하나로 변환하라.\n"
                "규칙:\n"
                "1. 반드시 합쇼체(-습니다, -ㅂ니다)만 사용. -다(평어)/-ㄴ다/-요/-해/-야 금지.\n"
                "2. 과거 표현 없으면 현재형 사용. 가다→갑니다(O) 갔습니다(X)\n"
                "3. 입력에 없는 단어(매일,아침,함께,항상 등) 추가 금지.\n"
                "4. 설명 없이 문장 하나만 출력.\n"
                "5. 반드시 한국어만 출력. 영어 단어 사용 금지."
            )},
            {"role": "user", "content": prompt},
        ]

    def _run_inference(self, prompt: str, do_sample: bool = False) -> str:
        import torch

        messages = self._build_messages(prompt)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

        gen_kwargs = {}
        if do_sample:
            gen_kwargs = {
                "do_sample": True,
                "temperature": self.gen_config.sampling_temperature,
                "top_p": self.gen_config.sampling_top_p,
            }

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(self, words: list[str], prompt: str = "") -> str:
        return self._run_inference(prompt, do_sample=False)

    def generate_with_sampling(self, words: list[str], prompt: str = "") -> str:
        return self._run_inference(prompt, do_sample=True)


# ---------------------------------------------------------------------------
# Backend 3: Fine-tuned EXAONE with LoRA adapter
# ---------------------------------------------------------------------------

class FinetunedGenerator:
    """Generate Korean sentences using fine-tuned EXAONE + LoRA."""

    ADAPTER_PATH = Path(__file__).resolve().parent / "exaone-finetuned-v4"
    MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    def __init__(self, adapter_path: str = "", model_id: str = "", gen_config: GenerationConfig | None = None, load_4bit: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        self.gen_config = gen_config or GenerationConfig()
        model_id = model_id or self.MODEL_ID
        adapter = adapter_path or str(self.ADAPTER_PATH)

        print(f"[sLLM] Loading fine-tuned adapter from {adapter} ({'4bit' if load_4bit else '16bit/bf16'}) ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = dict(device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
        if load_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        base = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        base.transformer.__class__.get_input_embeddings = lambda self: self.wte

        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()
        self.model.generation_config.max_new_tokens = self.gen_config.max_new_tokens_finetuned
        self.model.generation_config.do_sample = False
        self.model.generation_config.repetition_penalty = self.gen_config.repetition_penalty
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        print(f"[sLLM] Fine-tuned model loaded on {self.model.device}")

    def _build_messages(self, prompt: str, words: list[str]) -> list[dict]:
        return [
            {"role": "system", "content": (
                "단어 시퀀스를 자연스러운 한국어 문장 하나로 변환하라. "
                "반드시 합쇼체(-습니다, -ㅂ니다)만 사용하고, "
                "입력에 없는 단어를 추가하지 마라. "
                "반드시 한국어만 출력하고 영어 단어를 사용하지 마라."
            )},
            {"role": "user", "content": f"입력 단어: {prompt}" if prompt else f"입력 단어: {' / '.join(words)}"},
        ]

    def _run_inference(self, words: list[str], prompt: str, do_sample: bool = False) -> str:
        import torch

        messages = self._build_messages(prompt, words)
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

        gen_kwargs = {}
        if do_sample:
            gen_kwargs = {
                "do_sample": True,
                "temperature": self.gen_config.sampling_temperature,
                "top_p": self.gen_config.sampling_top_p,
            }

        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **gen_kwargs)

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def generate(self, words: list[str], prompt: str = "") -> str:
        return self._run_inference(words, prompt, do_sample=False)

    def generate_with_sampling(self, words: list[str], prompt: str = "") -> str:
        return self._run_inference(words, prompt, do_sample=True)


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class SLLMWordsToText:
    """7-layer controlled inference pipeline.

    Layer 1: Raw Generation
    Layer 2: Constraint Scoring
    Layer 3: Candidate Selection
    Layer 4: Format Cleanup
    Layer 5: Semantic Cleanup
    Layer 6: Final Guardrail
    Trace:   Inference tracing
    """

    def __init__(
        self,
        backend: str = "rule",
        gen_config: GenerationConfig | None = None,
        scoring_config: ScoringConfig | None = None,
        cleanup_config: CleanupConfig | None = None,
        adapter_path: str = "",
        model_id: str = "",
        load_4bit: bool = True,
    ):
        self.backend_name = backend
        self.gen_config = gen_config or GenerationConfig()
        self.scoring_config = scoring_config or ScoringConfig()
        self.cleanup_config = cleanup_config or CleanupConfig()

        # Initialize generator
        if backend == "exaone":
            self.generator = ExaoneGenerator(self.gen_config, model_id=model_id)
        elif backend == "finetuned":
            self.generator = FinetunedGenerator(adapter_path=adapter_path, model_id=model_id, gen_config=self.gen_config, load_4bit=load_4bit)
        else:
            self.generator = RuleBasedGenerator()

        # Initialize pipeline components
        self.scorer = ConstraintScorer(self.scoring_config)
        self.selector = CandidateSelector(self.scoring_config)
        self.format_cleaner = FormatCleaner(self.cleanup_config)
        self.semantic_cleaner = SemanticCleaner(self.cleanup_config)
        self.guardrail = FinalGuardrail()
        self.tracer = TraceRecorder(
            trace_dir=self.cleanup_config.trace_dir,
            enabled=self.cleanup_config.enable_tracing,
        )

    def normalize(self, words: list[str], request_id: str = "") -> str:
        """Run the full 7-layer pipeline.

        Input : list of Korean words
        Output: one normalized Korean sentence
        """
        start_time = time.time()
        trace = InferenceTrace(request_id=request_id, input_words=list(words))

        words = [str(w).strip() for w in words if str(w).strip()]
        if not words:
            return ""

        # Build prompt
        if self.gen_config.prompt_type == "basic":
            prompt = build_basic_prompt(words)
        else:
            prompt = build_fewshot_prompt(words)

        # === Layer 1: Raw Generation ===
        raw_candidates = []
        generation_methods = []

        # Greedy (deterministic)
        greedy_output = self.generator.generate(words, prompt=prompt)
        raw_candidates.append(greedy_output)
        generation_methods.append("greedy")

        # Sampling candidates
        if hasattr(self.generator, "generate_with_sampling"):
            for _ in range(self.gen_config.num_sampling_candidates):
                sampling_output = self.generator.generate_with_sampling(words, prompt=prompt)
                raw_candidates.append(sampling_output)
                generation_methods.append("sampling")

        trace.raw_candidates = list(raw_candidates)
        trace.generation_method = generation_methods

        # === Layer 2: Constraint Scoring ===
        scored = self.scorer.score_all(words, raw_candidates)
        trace.scores = [
            {
                "keyword": round(s.keyword_score, 3),
                "negation": round(s.negation_score, 3),
                "hallucination": round(s.hallucination_score, 3),
                "length": round(s.length_score, 3),
                "total": round(s.total_score, 3),
                "failures": [f.value for f in s.failure_types],
            }
            for s in scored
        ]

        # === Layer 3: Candidate Selection ===
        selection = self.selector.select(scored)
        trace.selected_index = raw_candidates.index(selection.best.raw_text) if selection.best.raw_text in raw_candidates else 0
        trace.fallback_index = raw_candidates.index(selection.fallback.raw_text) if selection.fallback and selection.fallback.raw_text in raw_candidates else -1
        trace.selection_reason = selection.selection_reason

        # === Layer 4: Format Cleanup (on best + fallback) ===
        best_formatted = self.format_cleaner.clean(selection.best.raw_text, input_words=words)
        trace.best_after_format_cleanup = best_formatted

        fallback_formatted = None
        if selection.fallback:
            fallback_formatted = self.format_cleaner.clean(selection.fallback.raw_text, input_words=words)
            trace.fallback_after_format_cleanup = fallback_formatted

        # === Layer 5: Semantic Cleanup (on best + fallback) ===
        best_semantic = self.semantic_cleaner.clean(best_formatted, input_words=words)
        trace.best_after_semantic_cleanup = best_semantic.text
        trace.cleanup_changes = [c.to_dict() for c in best_semantic.changes]

        fallback_cleaned = None
        if fallback_formatted:
            fallback_semantic = self.semantic_cleaner.clean(fallback_formatted, input_words=words)
            fallback_cleaned = fallback_semantic.text
            trace.fallback_after_semantic_cleanup = fallback_cleaned

        # === Layer 6: Final Guardrail ===
        guardrail_result = self.guardrail.check(
            best_cleaned=best_semantic.text,
            fallback_cleaned=fallback_cleaned,
            input_words=words,
        )
        trace.guardrail_passed = guardrail_result.passed
        trace.used_fallback = guardrail_result.used_fallback
        trace.guardrail_warnings = guardrail_result.warnings
        trace.final_output = guardrail_result.output

        # Collect all failure types
        all_failures = set()
        for s in scored:
            for f in s.failure_types:
                all_failures.add(f.value)
        for f in guardrail_result.failure_types:
            all_failures.add(f.value)
        trace.failure_types = sorted(all_failures)

        # === Trace ===
        trace.duration_ms = (time.time() - start_time) * 1000
        self.tracer.save(trace)

        return guardrail_result.output

    def get_model_name(self) -> str:
        if self.backend_name == "exaone":
            return ExaoneGenerator.MODEL_ID
        elif self.backend_name == "finetuned":
            return "EXAONE-3.5-2.4B-Instruct (LoRA fine-tuned)"
        return "rule-based-baseline"


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

SllmModule = SLLMWordsToText


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-sLLM: words -> one Korean sentence")
    parser.add_argument("words", nargs="*", help="Input words, e.g.  나 학교 오늘 가다")
    parser.add_argument("--input-file", help="Path to a JSON file containing a word list")
    parser.add_argument("--input-json", help="Inline JSON string")
    parser.add_argument("--prompt-type", choices=["basic", "fewshot"], default="fewshot")
    parser.add_argument("--backend", choices=["rule", "exaone", "finetuned"], default="rule",
                        help="rule = lookup table, exaone = base model, finetuned = QLoRA")
    parser.add_argument("--trace", action="store_true", help="Enable inference tracing")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    if args.input_json:
        data = json.loads(args.input_json)
        words = data.get("words", [])
        request_id = data.get("request_id", "")
    elif args.input_file:
        data = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            words = data.get("words", [])
            request_id = data.get("request_id", "")
        else:
            words = data
            request_id = ""
    else:
        words = args.words or []
        request_id = ""

    gen_config = GenerationConfig(prompt_type=args.prompt_type)
    cleanup_config = CleanupConfig(enable_tracing=args.trace)

    module = SLLMWordsToText(
        backend=args.backend,
        gen_config=gen_config,
        cleanup_config=cleanup_config,
    )
    result = module.normalize(words, request_id=request_id)
    print(result)


if __name__ == "__main__":
    main()
