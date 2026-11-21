# PR-MemRetriever

基于本地 Mem-T 研究实现的检索器项目。当前实现直接并行检索和配对评测。

- **串行基线**保留 Mem-T 的单调用提示词、工具、`finish(answer)` 和回退回答。
- **并行模式**在相同提示词主体后附加一段格式说明。一轮用 `<tool_calls>` 包住 1–2 个原格式的 `<tool_call>`。工具并发读取，观察按调用顺序合并并去重。单独的原格式调用仍可用。
- A/B 复用同一个快照、模型、评分和决策上限。并行模式的累计检索调用数也受该上限约束。评测将快照恢复到输出目录的独立数据库。

## 运行直接并行评测

在已有依赖的 Mem-T Linux 环境运行。按实际位置设置 `R200`（HF 导出权重）、`MEMORY`（已完成的验证记忆目录）、`BGE`（本地嵌入模型）和 `RUN_ROOT`（全新的输出路径）。测 R200 时使用其对应的 `legacy-v1` 快照。

```bash
common=(--config configs/memt4b.yaml
  --model "$R200" --memory "$MEMORY" --inference-engine vllm
  --set "memory.embedding_model=$BGE"
  --set retrieval.evidence_alpha=1 --set retrieval.expansion_rounds=1
  --limit 20)

python cli.py evaluate "${common[@]}" --parallel-calls 1 --output "$RUN_ROOT/serial"
python cli.py evaluate "${common[@]}" --parallel-calls 2 --output "$RUN_ROOT/parallel"
python -m evaluation.compare_parallel "$RUN_ROOT/serial" "$RUN_ROOT/parallel" \
  --output "$RUN_ROOT/comparison.json"
```

`--limit 20` 取固定题序的前 20 题，用来排查问题。去掉该参数可跑完整验证集。每组写出逐题预测、工具与结果 ID、F1、BLEU-1、回退率、调用数、决策数、token 及延迟。延迟含模型生成、工具、观察合并和必要的回退回答，从单题记忆快照准备完毕后开始计时。

要单独观察执行调度的影响，可增加一组 `--parallel-calls 2 --serial-tool-execution`，再与并行组配对。两组提示词和批内调用上限相同。用逐题日志确认实际动作匹配，重复测量后比较工具时间和端到端 P95。

## 本地检查与训练顺序

```bash
python -m pytest -q tests/test_parallel_retrieval.py
```

测试用 Mem-T 自带的小型词法库和脚本策略，覆盖工具重叠执行、确定性合并、源文本去重、格式与预算边界、单调用回归和 A/B 评测入口。真实结果须以 R200 和 BGE-M3 验证集的配对实测为准。

验证直接并行后，再接并行 MoT-GRPO rollout。先沿用原奖励短跑验收，随后独立比较 QA-only、冻结 Reader 和 Shapley。原训练入口与论文映射见 [TRAINING.md](TRAINING.md) 和 [ALGORITHM_MAP.md](docs/ALGORITHM_MAP.md)。上游版本及许可证见 [UPSTREAM.lock.json](UPSTREAM.lock.json) 和 [LICENSE](LICENSE)。

## 多轮并行对照

固定串行执行四次单检索，固定并行执行两轮双检索，两组共享四次调用预算。自适应组从双调用开始，后续依据观察选择一至两次检索或 `finish`。原 Mem-T 串行组提供端到端基线。四组共享一套 R200、验证记忆和推理引擎。

```bash
python -m evaluation.guided_parallel --model "$R200" --memory "$MEMORY" \
  --embedding "$BGE" --output "$RUN_ROOT/multiround"
```

使用两张 A100 和项目依赖中的 vLLM，结果写入 `report.json`，包括逐题调用、F1、P95 和真实完成至少两轮并行的题数。`--resume` 按已封存的逐题结果恢复，输入与源码必须匹配。

## 每轮调用上限对照

`python -m evaluation.guided_parallel --width-sweep` 使用相同的模型、快照和题目比较上限 2、4、8。首轮用满，各组都允许依据观察早停或再检索一轮，再回答。运行时传入上节的 `--model`、`--memory`、`--embedding` 和独立的 `--output`。报告并列记录 F1、实际调用数、独立记忆条目数和延迟。

## 全量 LoCoMo 三次重复

`evaluation.locomo_full prepare` 合并原 Mem-T 的 train、valid、test 记忆快照，并核对官方 LoCoMo 原始数据的 SHA-256。包含 10 段对话共 1,986 题，1,540 道常规题沿用原词面 F1，446 道对抗题单列拒答准确率。用 `--width-sweep --width-order 2 4 8` 运行一遍，另外两遍顺序为 `4 8 2`、`8 2 4`。`evaluation.locomo_full summarize` 校验三遍输入与逐题身份，输出按原划分和全量汇总的均值、样本标准差。
