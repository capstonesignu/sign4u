"""Fine-tune EXAONE-3.5 Instruct with QLoRA on honorific (합쇼체) data.

Parameterized (task #3 groundwork): MODEL_ID / DATA_PATH / OUTPUT_DIR and key
hyperparameters are CLI flags. Defaults reproduce the 2.4B v4 honorific retrain.

Switch to 7.8B via:
  python finetune_v3.py --model-id LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
      --output-dir exaone-finetuned-v4-7.8b --batch-size 1 --grad-accum 16

Calibration (measure throughput without full run):
  python finetune_v3.py --max-steps 20 --output-dir exaone-finetuned-v4-calib

Requirements: pip install torch transformers peft trl datasets accelerate bitsandbytes
"""
import argparse
import json
import os
from pathlib import Path

# Fix Windows encoding issue with trl jinja templates
os.environ["PYTHONUTF8"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


DEFAULT_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
DEFAULT_DATA_PATH = "train_data_all_final_chat.jsonl"
DEFAULT_OUTPUT_DIR = "./exaone-finetuned-v4"

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def parse_args():
    p = argparse.ArgumentParser(description="QLoRA fine-tune EXAONE on honorific data")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=160)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="override epochs for calibration (e.g. 20); -1 = use epochs")
    return p.parse_args()


def main(args):
    # ------------------------------------------------------------------
    # 1. 4-bit quantisation config (QLoRA) — fits 12 GB VRAM
    # ------------------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ------------------------------------------------------------------
    # 2. Load tokenizer + model
    # ------------------------------------------------------------------
    print(f"[finetune] Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[finetune] Loading model (4-bit NF4) from {args.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Enable gradient checkpointing to save VRAM
    model.gradient_checkpointing_enable()

    # Monkey-patch get_input_embeddings for PEFT compatibility
    model.transformer.__class__.get_input_embeddings = lambda self: self.wte

    # ------------------------------------------------------------------
    # 3. LoRA config
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ------------------------------------------------------------------
    # 4. Load dataset
    # ------------------------------------------------------------------
    print(f"[finetune] Loading dataset from {args.data_path} ...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    print(f"[finetune] {len(dataset)} total examples")

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_chat)

    # Train/validation split (90/10) to monitor overfitting
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"[finetune] Train: {len(train_dataset)}, Validation: {len(eval_dataset)}")

    # ------------------------------------------------------------------
    # 5. SFTConfig
    # ------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        max_length=args.max_length,
        dataset_text_field="text",
        max_steps=args.max_steps,
    )

    # ------------------------------------------------------------------
    # 6. SFT Trainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ------------------------------------------------------------------
    # 7. Train + save
    # ------------------------------------------------------------------
    eff = args.batch_size * args.grad_accum
    print("[finetune] Starting training ...")
    print(f"[finetune] Steps/epoch: {len(dataset) // eff}")
    print(f"[finetune] Total steps: {len(dataset) // eff * args.epochs}")
    trainer.train()
    print(f"[finetune] Saving model to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config for reference
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_id,
            "data": args.data_path,
            "data_entries_total": len(dataset),
            "train_entries": len(train_dataset),
            "eval_entries": len(eval_dataset),
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch": eff,
            "learning_rate": args.lr,
            "warmup_ratio": 0.05,
            "max_length": args.max_length,
            "quantization": "4bit NF4",
        }, f, indent=2)
    print("[finetune] Done!")


if __name__ == "__main__":
    main(parse_args())
