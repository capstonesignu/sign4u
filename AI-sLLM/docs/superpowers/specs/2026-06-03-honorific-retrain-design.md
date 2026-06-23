# 敬语整合 + EXAONE-2.4B 重训 — 设计文档 (Spec)

- **日期**: 2026-06-03
- **分支**: `honorific-v4`
- **仓库**: `D:\SLLM\AI-sLLM`
- **状态**: 设计已批准 → 待 spec 审核 → writing-plans → 执行

## 1. 背景

AI-sLLM 是韩国手语(KSL)翻译系统的**后处理 sLLM**:把 SLR 模型识别出的**词序列** → 一句**自然韩语**。
采用 7 层受控推理 pipeline,三个后端(`rule` / `exaone`-2.4B / `finetuned`-2.4B+QLoRA)。

组长(팀장)提出三项要求:
1. **任务#1** 每词位多候选 → 笛卡尔组合生成(组长**周二**细讲,**本 spec 不含**)
2. **任务#2** 말투 공손화: `~다`(평어) → `~습니다`(합쇼체) / `~어요`(해요체)
3. **任务#3** 换更大模型: EXAONE ≥7.8B,4bit 量化推理 < 1s(**本 spec 仅做参数化预埋,不实训 7.8B**)

MacBook 上已完成任务#2 的**代码**改动(MacBook 无法训练,故只改代码)并把数据敬语化。
本次在 RTX 5070 机器上**整合 + 重训**,真正落地任务#2,并为任务#3 预埋参数化。

> 말투决策:统一标准化为 **합쇼체(`~습니다`)** —— 组长允许 `~습니다` 或 `~어요`,二选一即可;
> 합쇼체 更正式、更契合医疗/医院场景,且与已敬语化的数据一致。

## 2. 现状(已核实)

| 资产 | `D:\SLLM\AI-sLLM`(主仓库) | `D:\sllm2026年6月3日`(快照) | MacBook 补丁 zip |
|---|---|---|---|
| 训练数据 chat.jsonl | 平语(7953 平 / 8 敬) | **敬语 8053 / 0 平** ✅ | — |
| 测试数据 test_clean.json | 平语(96) | **敬语 96** ✅ | — |
| 代码 pipeline | 旧 `POLITE_TO_PLAIN` | 旧 `POLITE_TO_PLAIN` | **`PLAIN_TO_POLITE`** ✅ |
| git / finetune 脚本 / adapter | ✅ (v3 / 5k / base) | ✗ | ✗ |

**结论**:没有任何位置同时具备「敬语数据 + 敬语代码」。整合 = 把敬语数据 + 敬语代码补丁都汇入主仓库,再重训。

**环境**: Python 3.11.9 / torch 2.11.0+cu128 / transformers 5.8.0 / peft / bitsandbytes;
GPU RTX 5070 12GB;base 模型 `EXAONE-3.5-2.4B-Instruct` 已缓存于 `D:\hf_cache`(**无需下载**)。
git 当前在 `honorific-v4`,tracked 文件干净(仅 18 个未跟踪的 `scripts/*.py`,不影响,不纳入本次提交)。

## 3. 范围

- **In**: ① 敬语数据整合　② 敬语代码补丁(5 文件)　③ finetune/inference 参数化(任务#3 预埋)　④ 重训 2.4B + 三阶段验证。
- **Out**: 7.8B 实训(任务#3 后续)、任务#1(笛卡尔生成)、数据本身的再生成。

## 4. 详细设计

### ① 数据整合
把快照敬语数据搬入主仓库,旧平语版另存 `*.plain.bak`:
- `train_data_all_final_chat.jsonl`(8072 条,finetune 实际使用)
- `train_data_all_final.json`、`test_data_clean.json`(96 条,evaluate 使用)
- 复查 chat 中 19 条"非标准结尾"(疑问句/特殊收尾),仅记录、不阻断。
- 整合前核对:快照与主仓库条数一致(均 8072 / 96),确保是 1:1 敬语化版本。

### ② 代码补丁(覆盖前先 `git diff` 确认仅敬语相关改动)
| 文件 | 改动要点 |
|---|---|
| `format_cleanup.py` | `POLITE_TO_PLAIN` → **`PLAIN_TO_POLITE`**(평어/해요체/honorific → 합쇼체);`_polite_to_plain`→`_plain_to_polite` |
| `semantic_cleanup.py` | Layer5 `PAST_TO_PRESENT` / `GRAMMAR_FIXES` 目标全部改 합쇼체 |
| `prompt_templates.py` | `SYSTEM_INSTRUCTION` 规则1 + 20 条 few-shot 改 합쇼체 |
| `sllm_module.py` | `ExaoneGenerator` / `FinetunedGenerator` 两个 system prompt 改 합쇼체 |
| `config/cleanup_config.py` | `enable_polite_to_plain` → `enable_plain_to_polite`(默认 `True`) |

### ③ 参数化(任务#3 预埋,行为默认不变)
- `finetune_v3.py`: `MODEL_ID / DATA_PATH / OUTPUT_DIR` 改 `argparse`,默认值 = 现状(2.4B / 新数据 / `exaone-finetuned-v4`)。
  切 7.8B 仅需:`--model-id LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --output-dir exaone-finetuned-v4-7.8b`。
- `sllm_module.py`: `FinetunedGenerator` 的 `ADAPTER_PATH` 与底座 `MODEL_ID` 可由构造参数/CLI 传入,默认指向 `exaone-finetuned-v4`。

### ④ 重训
- 从 **base 2.4B** 起(**不**续 v3,避免带入平语习惯),QLoRA 4bit NF4,r=16 / α=32,4 epoch,lr=8e-5,bs=2×accum8,max_len=160(沿用 `finetune_v3` 现有超参)。
- 输出 → `exaone-finetuned-v4`(保留旧 `exaone-finetuned-v3`)。

## 5. 验证(pass 标准)

- **Phase 1(无需 GPU)**: 对 CHANGES 的 12 句 + test 集抽样跑 Layer4+5 → **100% 합쇼체**;`pytest tests/` 通过。
- **Phase 2(训练)**: train/val loss 正常下降、无 OOM、val 不发散。
- **Phase 3(评估)**: 新 adapter 跑 `evaluate.py` 对敬语 test 集 →
  - **敬语率 ≈ 100%**(关键指标)
  - BLEU 不显著低于旧 v3(参考已变敬语,允许小幅波动)
  - 记录单句推理延迟(4bit),作为任务#3 `<1s` 的基准。

## 6. 风险与缓解
- **OOM**: 2.4B QLoRA 在 12GB 上之前训过,低风险;若紧 → bs=1 / accum=16。
- **覆盖误删**: 补丁覆盖前逐个 `git diff`;全程在 `honorific-v4` 分支。
- **数据-代码不一致**: 整合前核对条数(8072 / 96)。
- **Layer4/5 冗余**: 敬语模型之后退化为幂等"安全网",对已敬语文本无害。

## 7. 回滚
全程在 `honorific-v4` 分支;旧数据存 `*.plain.bak`;旧 adapter `exaone-finetuned-v3` 原样保留。
放弃改动直接 `git checkout main`。

## 8. 时间预估(2.4B)
| 阶段 | 预估 |
|---|---|
| ①②③ + Phase 1(无 GPU) | ~30–45 min |
| ④ 训练(QLoRA 4 epoch,base 已缓存) | ~1–2 h(开训后给精确 ETA) |
| Phase 3 评估 + 测速 | ~10–20 min |
| **合计** | best ~1.5 h / 正常 ~2–2.5 h(大头是无人值守训练) |
