# 敬语整合 + EXAONE-2.4B 重训 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把敬语数据 + 敬语代码补丁汇入主仓库 `D:\SLLM\AI-sLLM`,重训 EXAONE-2.4B 使其原生输出 합쇼체,并为任务#3(7.8B)预埋参数化。

**Architecture:** 7 层受控推理 pipeline 不变。改动:① 替换为敬语训练/测试数据 ② 覆盖 5 个 honorific 代码补丁文件(Layer4 `PLAIN_TO_POLITE` 等)③ `finetune_v3.py`/`FinetunedGenerator` 参数化(`MODEL_ID`/`OUTPUT_DIR`/`ADAPTER_PATH`)④ QLoRA 4bit 重训出 `exaone-finetuned-v4`。

**Tech Stack:** Python 3.11 / torch 2.11+cu128 / transformers 5.8 / peft / trl / bitsandbytes;GPU RTX 5070 12GB;base 模型已缓存于 `D:\hf_cache`。

**关键路径常量:**
- 主仓库: `D:\SLLM\AI-sLLM`(当前分支 `honorific-v4`)
- 补丁源: `C:\Users\User\Desktop\sllm_macbook_patch`
- 敬语数据源: `D:\sllm2026年6月3日`
- 数据核查脚本: `C:\Users\User\AppData\Local\Temp\check_data.py`

**所有命令默认在仓库根目录执行**;每条 PowerShell 命令前先:
```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
```

---

## Task 1: 敬语数据整合 (①)

**Files:**
- Backup(本地, gitignore): `train_data_all_final_chat.jsonl.plain.bak`, `train_data_all_final.json.plain.bak`, `test_data_clean.json.plain.bak`
- Overwrite(git tracked): `train_data_all_final_chat.jsonl`, `train_data_all_final.json`, `test_data_clean.json`

- [ ] **Step 1: 确认这 3 个数据文件当前被 git 跟踪(作为兜底备份)**

```powershell
cd D:\SLLM\AI-sLLM
git ls-files train_data_all_final_chat.jsonl train_data_all_final.json test_data_clean.json
```
Expected: 三个文件名都列出(说明已 tracked,旧平语版在 git 历史里可恢复)。

- [ ] **Step 2: 本地备份当前(平语)版本为 `*.plain.bak`**

```powershell
cd D:\SLLM\AI-sLLM
Copy-Item train_data_all_final_chat.jsonl train_data_all_final_chat.jsonl.plain.bak -Force
Copy-Item train_data_all_final.json       train_data_all_final.json.plain.bak       -Force
Copy-Item test_data_clean.json            test_data_clean.json.plain.bak            -Force
```
Expected: 无报错。

- [ ] **Step 3: 用敬语版覆盖(从快照拷贝)**

```powershell
cd D:\SLLM\AI-sLLM
$src = "D:\sllm2026年6月3日"
Copy-Item "$src\train_data_all_final_chat.jsonl" .\train_data_all_final_chat.jsonl -Force
Copy-Item "$src\train_data_all_final.json"       .\train_data_all_final.json       -Force
Copy-Item "$src\test_data_clean.json"            .\test_data_clean.json            -Force
```
Expected: 无报错。

- [ ] **Step 4: 核查敬语率(用已写好的 check_data.py)**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
python C:\Users\User\AppData\Local\Temp\check_data.py
```
Expected(关键行): `CHAT D:/SLLM: honorific=8053 plain=0 other=19` 且 `JSON test D:/SLLM: n=96 honorific=96 plain=0`。
若 chat 仍显示 plain=7953 → 拷贝没生效,停止排查。

- [ ] **Step 5: 提交敬语数据**

```powershell
cd D:\SLLM\AI-sLLM
git add train_data_all_final_chat.jsonl train_data_all_final.json test_data_clean.json
git commit -m "data: replace train/test data with honorific (합쇼체) version

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected: `3 files changed`。(`.plain.bak` 不提交 —— Task 3 会把它加进 .gitignore;现在它们只是未跟踪文件。)

---

## Task 2: 敬语代码补丁 + TDD 测试 (②)

**Files:**
- Test (create): `tests/test_honorific.py`
- Overwrite: `format_cleanup.py`, `semantic_cleanup.py`, `prompt_templates.py`, `sllm_module.py`, `config/cleanup_config.py`

- [ ] **Step 1: 写失败测试 `tests/test_honorific.py`**

```python
"""Honorific (합쇼체) normalization tests for Layer 4 Format Cleanup.

After the honorific patch, FormatCleaner must normalize 평어/해요체 endings
into 합쇼체 (-습니다/-ㅂ니다), and must be idempotent on already-합쇼체 text.
Cases verified empirically against the patched format_cleanup.
"""
import sys
import os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from config import CleanupConfig
from format_cleanup import FormatCleaner


# 평어/해요체 -> 합쇼체 (each value verified against patched code)
PLAIN_TO_POLITE_CASES = [
    ("간다.", "갑니다."),
    ("필요하다.", "필요합니다."),
    ("마셔요.", "마십니다."),
    ("기다린다.", "기다립니다."),
    ("가니?", "갑니까?"),
    ("중이다.", "중입니다."),
    ("좋다.", "좋습니다."),
    ("많다.", "많습니다."),
    ("아프다.", "아픕니다."),
    ("한다.", "합니다."),
    ("온다.", "옵니다."),
    ("먹는다.", "먹습니다."),
]

# already 합쇼체 -> unchanged (idempotency; OLD code would mangle these)
IDEMPOTENT_CASES = ["갑니다.", "필요합니다.", "먹습니다.", "좋습니다."]


@pytest.fixture
def cleaner():
    return FormatCleaner(CleanupConfig())


@pytest.mark.parametrize("plain,polite", PLAIN_TO_POLITE_CASES)
def test_plain_to_polite(cleaner, plain, polite):
    assert cleaner.clean(plain) == polite


@pytest.mark.parametrize("text", IDEMPOTENT_CASES)
def test_idempotent_on_polite(cleaner, text):
    assert cleaner.clean(text) == text
```

- [ ] **Step 2: 运行测试,确认在旧代码上失败**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
pytest tests/test_honorific.py -v
```
Expected: FAIL。旧代码 `간다.`→`간다.`(≠`갑니다.`)、`필요합니다.`→`필요한다.`(≠`필요합니다.`)。多条 assert 失败。

- [ ] **Step 3: 覆盖 5 个补丁文件**

```powershell
cd D:\SLLM\AI-sLLM
$p = "C:\Users\User\Desktop\sllm_macbook_patch"
Copy-Item "$p\format_cleanup.py"   .\format_cleanup.py   -Force
Copy-Item "$p\semantic_cleanup.py" .\semantic_cleanup.py -Force
Copy-Item "$p\prompt_templates.py" .\prompt_templates.py -Force
Copy-Item "$p\sllm_module.py"      .\sllm_module.py      -Force
Copy-Item "$p\cleanup_config.py"   .\config\cleanup_config.py -Force
```
Expected: 无报错。(注意 `cleanup_config.py` 落到 `config\` 子目录。)

- [ ] **Step 4: `git diff` 确认每个文件只有敬语相关改动**

```powershell
cd D:\SLLM\AI-sLLM
git diff --stat
git diff config/cleanup_config.py
```
Expected:
- `config/cleanup_config.py`: 仅 `enable_polite_to_plain` → `enable_plain_to_polite`。
- `format_cleanup.py`: `POLITE_TO_PLAIN`→`PLAIN_TO_POLITE` 表 + `_polite_to_plain`→`_plain_to_polite` + docstring。
- `semantic_cleanup.py`: `PAST_TO_PRESENT`/`GRAMMAR_FIXES` 目标变 합쇼체。
- `prompt_templates.py`: system rule1 + few-shot 变 합쇼체。
- `sllm_module.py`: 两个 generator 的 system prompt 变 합쇼체(`-습니다`)。
若出现与敬语无关的删改(如丢失某函数)→ 停止,手动核对。

- [ ] **Step 5: 运行 honorific 测试,确认通过**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
pytest tests/test_honorific.py -v
```
Expected: PASS(16 passed)。

- [ ] **Step 6: 跑全量回归测试**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
pytest tests/ -v
```
Expected: 全部 PASS(含 `test_constraints.py`、`test_postprocess.py`、`test_honorific.py`)。若旧测试因敬语改动而失败,记录并判断是否是预期的(平语断言需更新);仅在确属过时断言时更新对应测试。

- [ ] **Step 7: 提交**

```powershell
cd D:\SLLM\AI-sLLM
git add tests/test_honorific.py format_cleanup.py semantic_cleanup.py prompt_templates.py sllm_module.py config/cleanup_config.py
git commit -m "feat: apply honorific (합쇼체) cleanup patch + tests

Layer4 PLAIN_TO_POLITE, Layer5 합쇼체 targets, honorific prompts.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected: `6 files changed`。

---

## Task 3: 参数化(任务#3 预埋)(③)

**Files:**
- Modify: `finetune_v3.py`(加 argparse)
- Modify: `sllm_module.py:219-262`(`FinetunedGenerator` 默认 adapter→v4 + `model_id` 参数;`ExaoneGenerator` 加 `model_id` 参数)
- Modify: `.gitignore`(加 `exaone-finetuned-v4/`、`*.plain.bak`)

- [ ] **Step 1: 参数化 `finetune_v3.py`** —— 把模块常量 + `main()` 改成 argparse 驱动

把文件顶部的三个常量定义:
```python
MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
DATA_PATH = "train_data_all_final_chat.jsonl"
OUTPUT_DIR = "./exaone-finetuned-v3"
```
替换为:
```python
import argparse

DEFAULT_MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
DEFAULT_DATA_PATH = "train_data_all_final_chat.jsonl"
DEFAULT_OUTPUT_DIR = "./exaone-finetuned-v4"


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
```

把 `def main():` 改为 `def main(args):`,并将函数体内引用替换:
- `MODEL_ID` → `args.model_id`
- `DATA_PATH` → `args.data_path`
- `OUTPUT_DIR` → `args.output_dir`

在 `SFTConfig(...)` 里把硬编码超参替换为 args,并新增 `max_steps`:
```python
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
```
把底部:
```python
if __name__ == "__main__":
    main()
```
改为:
```python
if __name__ == "__main__":
    main(parse_args())
```
(同时把 `training_config.json` dump 里的字面量 `MODEL_ID`/`DATA_PATH` 改成 `args.model_id`/`args.data_path`,`epochs`/`batch_size` 等改成对应 args。)

- [ ] **Step 2: 参数化 `sllm_module.py` 的 `FinetunedGenerator`**

把(补丁后的)`FinetunedGenerator` 头部:
```python
class FinetunedGenerator:
    """Generate Korean sentences using fine-tuned EXAONE-3.5-2.4B-Instruct + LoRA."""

    ADAPTER_PATH = Path(__file__).resolve().parent / "exaone-finetuned-v3"

    def __init__(self, adapter_path: str = "", gen_config: GenerationConfig | None = None):
```
改为:
```python
class FinetunedGenerator:
    """Generate Korean sentences using fine-tuned EXAONE + LoRA."""

    ADAPTER_PATH = Path(__file__).resolve().parent / "exaone-finetuned-v4"
    MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    def __init__(self, adapter_path: str = "", model_id: str = "", gen_config: GenerationConfig | None = None):
```
并把其 `__init__` 内:
```python
        model_id = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
        adapter = adapter_path or str(self.ADAPTER_PATH)
```
改为:
```python
        model_id = model_id or self.MODEL_ID
        adapter = adapter_path or str(self.ADAPTER_PATH)
```

- [ ] **Step 3: 参数化 `sllm_module.py` 的 `ExaoneGenerator`(为 7.8B 预埋,行为不变)**

把:
```python
class ExaoneGenerator:
    """Generate Korean sentences using EXAONE-3.5-2.4B-Instruct."""

    MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    def __init__(self, gen_config: GenerationConfig | None = None):
```
改为:
```python
class ExaoneGenerator:
    """Generate Korean sentences using EXAONE-3.5 Instruct."""

    MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

    def __init__(self, gen_config: GenerationConfig | None = None, model_id: str = ""):
        self._model_id = model_id or self.MODEL_ID
```
并把 `__init__` 内 `self.MODEL_ID` 的两处使用(`AutoTokenizer.from_pretrained(self.MODEL_ID...)` 与 `AutoModelForCausalLM.from_pretrained(self.MODEL_ID...)`)改为 `self._model_id`。
**注意**:`get_model_name()` 里仍引用 `ExaoneGenerator.MODEL_ID`(类属性),保持不变即可。

- [ ] **Step 4: 更新 `.gitignore`**

在 `.gitignore` 的 adapter 段(`exaone-finetuned-v3/` 那几行下面)追加:
```
exaone-finetuned-v4/
exaone-finetuned-v4-7.8b/
*.plain.bak
exaone-finetuned-v4-calib/
```

- [ ] **Step 5: 冒烟测试 —— argparse 与 import 正常**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
python finetune_v3.py --help
python -c "import ast; ast.parse(open('sllm_module.py',encoding='utf-8').read()); print('sllm_module OK')"
python -c "import ast; ast.parse(open('finetune_v3.py',encoding='utf-8').read()); print('finetune_v3 OK')"
```
Expected: `--help` 列出 `--model-id/--data-path/--output-dir/--max-steps` 等;两个 `... OK`。

- [ ] **Step 6: 提交**

```powershell
cd D:\SLLM\AI-sLLM
git add finetune_v3.py sllm_module.py .gitignore
git commit -m "feat: parameterize MODEL_ID/OUTPUT_DIR/ADAPTER for task #3 (default v4)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected: `3 files changed`。

---

## Task 4: Phase 1 端到端无 GPU 验证

**Files:** 无(仅运行)

- [ ] **Step 1: 用 rule 后端跑 pipeline,确认端到端输出 합쇼체**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
python sllm_module.py 나 학교 오늘 가다 --backend rule
python sllm_module.py 도움 필요 --backend rule
python sllm_module.py 진료 예약 하다 --backend rule
```
Expected: 每句以 합쇼체 结尾,例如 `나는 오늘 학교에 갑니다.`、`도움이 필요합니다.`、`진료를 예약합니다.`(末尾为 `…습니다.`/`…ㅂ니다.`)。

- [ ] **Step 2: 全量测试复跑(确保 Task 3 改动未破坏)**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
pytest tests/ -q
```
Expected: 全 PASS。无需提交(无文件改动)。

---

## Task 5: 重训 2.4B (④) + Phase 2

**Files:** Create(gitignored): `exaone-finetuned-v4/`

- [ ] **Step 1: 校准跑 20 步,测吞吐 + 确认不 OOM**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
python finetune_v3.py --max-steps 20 --output-dir exaone-finetuned-v4-calib
```
Expected: 加载 base(从 `D:\hf_cache`,无下载)→ 训练进度条显示 `it/s` 与 ETA → 20 步后保存到 `exaone-finetuned-v4-calib`。
记录 `it/s`,用它估算满训 ETA:总优化步 ≈ `7265 / (batch 2 × accum 8) × 4 epoch ≈ 1816 步`;
满训时长 ≈ `1816 / it/s`(注意每优化步 = 8 次 micro-batch,进度条的 it 通常是优化步)。
若 OOM → 改用 `--batch-size 1 --grad-accum 16` 重试本步。

- [ ] **Step 2: 删除校准产物**

```powershell
cd D:\SLLM\AI-sLLM
Remove-Item exaone-finetuned-v4-calib -Recurse -Force -ErrorAction SilentlyContinue
```

- [ ] **Step 3: 启动满量训练(后台,沿用默认超参)**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
python finetune_v3.py 2>&1 | Tee-Object -FilePath train_v4.log
```
(用 `run_in_background: true` 运行此命令。)
Expected: 训练启动,`train_v4.log` 持续写入 loss。base 已缓存,默认即 2.4B/新数据/`exaone-finetuned-v4`/4 epoch。
**若 Step 1 测得 ETA 明显偏长(>3h),先回报用户再决定是否启动。**

- [ ] **Step 4: 监控训练直到完成**

通过读取 `train_v4.log` 观察:train loss 应稳定下降;每 epoch 末打印 eval loss,不应持续上升(过拟合)。
Expected(完成时): 日志出现 `[finetune-v3] Done!`,且 `exaone-finetuned-v4/` 下有 `adapter_model.safetensors`、`adapter_config.json`、`training_config.json`。

```powershell
cd D:\SLLM\AI-sLLM
Get-ChildItem exaone-finetuned-v4 | Select-Object Name, Length
```

---

## Task 6: Phase 3 评估 + 测速 + 收尾

**Files:**
- Create(gitignored): `evaluation_results_v4.json`
- Create: `eval_honorific_check.py`(临时校验脚本,可放 `scripts/`)
- Modify: `CHANGES_敬语化补丁.md` 或 `README.md`(记录结果)

- [ ] **Step 1: 用新 adapter 跑评估**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
python evaluate.py --backend finetuned --data test_data_clean.json --output evaluation_results_v4.json
```
Expected: 跑完 96 条,生成 `evaluation_results_v4.json`(含 BLEU 等)。`finetuned` 后端默认加载 `exaone-finetuned-v4`(Task 3 已设默认)。

- [ ] **Step 2: 校验输出敬语率 + 测单句延迟**

创建 `scripts/eval_honorific_check.py`:
```python
# -*- coding: utf-8 -*-
import json, io, sys, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

res = json.load(open("evaluation_results_v4.json", encoding="utf-8"))
rows = res["rows"]                       # 确认结构: evaluate() 返回 {"rows":[{"prediction","reference",...}]}
preds = [r["prediction"] for r in rows]
print(f"BLEU={res['avg_sentence_bleu']:.4f}  chrF++={res['avg_chrf_plusplus']:.4f}  "
      f"kw_recall={res['avg_keyword_recall']:.2%}  exact={res['exact_match']:.2%}")
def is_hon(t):
    t = t.rstrip()
    return t.endswith(("습니다.", "ㅂ니다.", "습니까?", "ㅂ니까?", "니다.", "니까?"))
hon = sum(1 for p in preds if is_hon(p))
print(f"predictions={len(preds)}  honorific={hon}  rate={hon/max(1,len(preds)):.1%}")
for p in preds[:8]:
    print("   -", p)
```
```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
python scripts/eval_honorific_check.py
```
Expected: 敬语率 ≥ 95%(目标 ~100%);同时打印 BLEU/chrF++(供 Step 4 记录)。

- [ ] **Step 3: 测单句推理延迟(4bit,记录为任务#3 基准)**

```powershell
cd D:\SLLM\AI-sLLM ; $env:PYTHONUTF8 = "1"
python -c "import time,sys,io; sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8'); from sllm_module import SLLMWordsToText as M; m=M(backend='finetuned'); m.normalize(['나','학교','오늘','가다']); t=time.time(); [m.normalize(['진료','예약','하다']) for _ in range(5)]; print('avg sec/sentence:', round((time.time()-t)/5,3))"
```
Expected: 打印平均每句秒数(记录;2.4B 4bit 预期 < 1s,作为 7.8B 的对照基准)。

- [ ] **Step 4: 记录结果到 CHANGES,提交代码改动**

在 `CHANGES_敬语化补丁.md` 末尾(或 `README.md`)追加一节"v4 重训结果":敬语率、BLEU、平均延迟、adapter=`exaone-finetuned-v4`。
```powershell
cd D:\SLLM\AI-sLLM
git add scripts/eval_honorific_check.py CHANGES_敬语化补丁.md
git commit -m "eval: v4 honorific retrain results + honorific-rate check

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
Expected: 提交成功(`evaluation_results_v4.json` 被 .gitignore 忽略,不提交)。

- [ ] **Step 5: 收尾**

调用 `superpowers:finishing-a-development-branch` 决定 `honorific-v4` 的合并/PR/清理方式。汇报:敬语率、延迟、BLEU、训练时长,以及任务#3 一键切 7.8B 的命令:
```
python finetune_v3.py --model-id LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --output-dir exaone-finetuned-v4-7.8b --batch-size 1 --grad-accum 16
```

---

## 验证标准对照(Spec §5)

| Phase | 标准 | 对应 Task |
|---|---|---|
| Phase 1(无 GPU) | 12 句 + tests 全绿;rule 端到端输出 합쇼체 | Task 2 Step5-6, Task 4 |
| Phase 2(训练) | loss 正常下降、无 OOM、val 不发散 | Task 5 Step1,4 |
| Phase 3(评估) | 敬语率 ~100%;记录 BLEU + 延迟 | Task 6 Step1-3 |

## 回滚
全程在 `honorific-v4` 分支;放弃改动 `git checkout main`;旧 adapter `exaone-finetuned-v3` 与 `*.plain.bak` 原样保留。
