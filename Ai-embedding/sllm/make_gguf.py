"""local_sLLM/base_model + local_sLLM/adapter → local_sLLM/model_q4_k_m.gguf

Usage:
    cd local_sLLM
    python make_gguf.py
"""

import subprocess
import sys
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR    = Path(__file__).parent
BASE_MODEL  = BASE_DIR / "base_model"
ADAPTER     = BASE_DIR / "adapter"
MERGED_DIR  = BASE_DIR / "merged_tmp"
GGUF_FP16   = MERGED_DIR / "model_fp16.gguf"
GGUF_Q4     = BASE_DIR / "model_q4_k_m.gguf"
LLAMACPP    = Path(__file__).parent.parent.parent / "llama.cpp"
CONVERT     = LLAMACPP / "convert_hf_to_gguf.py"


def _patch_exaone(model):
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        transformer.__class__.get_input_embeddings = lambda self: self.wte
        transformer.__class__.set_input_embeddings = lambda self, v: setattr(self, "wte", v)
    try:
        model.get_output_embeddings()
    except (NotImplementedError, AttributeError):
        model.__class__.get_output_embeddings = lambda self: getattr(self, "lm_head", None)
        model.__class__.set_output_embeddings = lambda self, v: setattr(self, "lm_head", v)


def step1_merge():
    print(f"\n[1/3] Merge  {BASE_MODEL} + {ADAPTER}")
    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    _patch_exaone(base)
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    model = model.merge_and_unload()
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"[1/3] Done → {MERGED_DIR}")


def step2_convert():
    print(f"\n[2/3] Convert HF → GGUF fp16")
    if not CONVERT.exists():
        print(f"ERROR: {CONVERT} not found", file=sys.stderr)
        sys.exit(1)
    subprocess.run(
        [sys.executable, str(CONVERT), str(MERGED_DIR),
         "--outfile", str(GGUF_FP16), "--outtype", "f16"],
        check=True,
    )
    print(f"[2/3] Done → {GGUF_FP16}")


def step3_quantize():
    print(f"\n[3/3] Quantize fp16 → Q4_K_M")
    quantize = shutil.which("llama-quantize") or "/opt/homebrew/bin/llama-quantize"
    subprocess.run(
        [quantize, str(GGUF_FP16), str(GGUF_Q4), "Q4_K_M"],
        check=True,
    )
    size_gb = GGUF_Q4.stat().st_size / 1024**3
    print(f"[3/3] Done → {GGUF_Q4}  ({size_gb:.2f} GB)")


def cleanup():
    print(f"\n[cleanup] Removing {MERGED_DIR}")
    shutil.rmtree(MERGED_DIR, ignore_errors=True)
    fp16 = MERGED_DIR / "model_fp16.gguf"
    if fp16.exists():
        fp16.unlink()


if __name__ == "__main__":
    step1_merge()
    step2_convert()
    step3_quantize()
    cleanup()
    print(f"\n완료: {GGUF_Q4}")
