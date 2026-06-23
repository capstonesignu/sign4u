"""Generate human evaluation sheet using the new 7-layer pipeline.

Runs finetuned backend on test_data_clean.json (96 samples),
outputs a plain-text evaluation form.
"""
import sys, io, json, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from sllm_module import SLLMWordsToText
from config import GenerationConfig, ScoringConfig, CleanupConfig

# Load test data
data_path = Path("test_data_clean.json")
samples = json.loads(data_path.read_text(encoding="utf-8"))
print(f"Loaded {len(samples)} samples")

# Initialize pipeline with tracing
cleanup_cfg = CleanupConfig(enable_tracing=True, trace_dir="traces_v2")
module = SLLMWordsToText(backend="finetuned", cleanup_config=cleanup_cfg)

# Run inference
results = []
start = time.time()
for i, sample in enumerate(samples):
    words = sample["words"]
    ref = sample.get("reference", "")
    sid = sample.get("id", f"sample-{i+1:03d}")

    output = module.normalize(words, request_id=sid)
    results.append({
        "id": sid,
        "words": words,
        "reference": ref,
        "system_output": output,
    })
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start
        print(f"  [{i+1}/{len(samples)}] {elapsed:.1f}s elapsed")

elapsed = time.time() - start
print(f"Done: {len(results)} samples in {elapsed:.1f}s")

# Save results JSON
results_path = Path("human_eval_v2_results.json")
results_path.write_text(
    json.dumps(results, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

# Generate plain-text evaluation form
lines = []
lines.append("=" * 60)
lines.append("한국수어 번역 시스템 — 인간 평가서 (v2: 7-Layer Pipeline)")
lines.append("=" * 60)
lines.append("")
lines.append("모델: EXAONE-3.5-2.4B-Instruct (QLoRA fine-tuned)")
lines.append("파이프라인: 7-Layer Controlled Inference Pipeline")
lines.append(f"샘플 수: {len(results)}")
lines.append(f"생성 시간: {elapsed:.1f}초")
lines.append("")
lines.append("-" * 60)
lines.append("평가 기준 (1~5점)")
lines.append("-" * 60)
lines.append("자연스러움: 한국어 원어민이 읽었을 때 자연스러운 문장인가?")
lines.append("문법 정확성: 조사, 어미, 시제, 활용이 올바른가?")
lines.append("의미 충실도: 입력 단어들의 의미가 출력에 정확히 반영되었는가?")
lines.append("정보 정확성: 입력에 없는 내용을 추가(환각)하지 않았는가?")
lines.append("")
lines.append("=" * 60)
lines.append("평가표")
lines.append("=" * 60)

for r in results:
    lines.append("")
    lines.append(f"[{r['id']}]")
    lines.append(f"  입력 단어: {' / '.join(r['words'])}")
    lines.append(f"  참고 문장: {r['reference']}")
    lines.append(f"  시스템 출력: {r['system_output']}")
    lines.append(f"  자연스러움:___  문법:___  충실도:___  정확성:___  메모:")

lines.append("")
lines.append("=" * 60)
lines.append("평가자 정보")
lines.append("이름:              날짜:              서명:")
lines.append("=" * 60)

output_path = Path("human_eval_v2.txt")
output_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Evaluation form saved to: {output_path}")
