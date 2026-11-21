# AgentMemory 六方法复现说明

本工作区统一运行 RecMem、FullContext、Naive RAG、MemoryOS、Mem0 OSS 和 A-Mem 的 LoCoMo／LongMemEval-S 实验。目录已同步到远程个人目录；下文保留可重复使用的迁移、运行命令。远程所有操作限于 `/public/home/wwxia/yuang.li`，模型权重仍在其 `models/`，不碰其他用户的文件、环境或进程。

## 目录与职责

```text
AgentMemory/
├── methods/
│   ├── RecMem/          原仓库、共用数据加载/评分/汇总/迁移入口
│   ├── MemoryOS/        上游源码及本地端点适配
│   ├── mem0/            Mem0 OSS 上游仓库（含 evaluation 子模块）
│   ├── A-mem/           当前下载的记忆库源码
│   ├── fullcontext/     直接使用近期对话窗口回答
│   └── naive_rag/       对话块向量检索后回答
├── artifacts/
│   ├── data/            LoCoMo、LongMemEval-S 数据
│   ├── results/         答案、评分、汇总及运行清单
│   ├── logs/            服务和实验日志
│   ├── stores/          六方法隔离的向量/记忆存储
│   └── cache/           NLTK、pip 等缓存
├── docs/references/     论文 PDF 和参考文本
├── .env                 本机私有变量；不提交、不展示密钥
├── .env.example         无密钥示例
└── AGENTS.md            共享节点工作边界
```

`methods/RecMem/evaluation/run_experiments.py` 运行前三种方法及统一评分；`external_runner.py` 只导入当前选中的外部方法；`compare_results.py` 逐题核对六份答案/评分后汇总。FullContext、Naive RAG 的类在各自目录，共用提示词和上下文预算仍在 `evaluation/baselines.py`。各上游仓库保留自己的 `.git` 与内部文件，不新建顶层 Git 仓库。A-Mem 属于**记忆库适配实验**，不是其缺失的论文评测仓库原版结果；MemoryOS 的 LongMemEval-S 也是新增适配。

六方法使用同一数据文件、同一生成/embedding 服务、同一问题及评分入口。LoCoMo 只评第 1–4 类；LongMemEval-S 默认全量，小样本按类别固定种子 42 抽样。标准答案和证据仅进入评分；生成答案 JSON 的行字段为 `question`、`answer`、`category`、`response`、`answerable`。`*.run.json` 记录数据哈希、参数与源码哈希；`*.context.json` 记录适用方法的上下文截断。FullContext 超窗时保留近期历史，因此应标记为 windowed FullContext，而不是无损全上下文。

## 远程环境与数据（完成目录同步和历史迁移后）

先按 [`.env.example`](.env.example) 检查远程根目录 `.env`：`PYTHONPATH` 必须同时包含 `AgentMemory` 和 `methods/RecMem`；数据、评分、存储、NLTK 路径应指向 `artifacts/`。已有 `.env` 不要用示例文件覆盖，先核对自己的旧值。Bash 加载后可用 `printf '%s\n' "$PYTHONPATH" "$LOCOMO_PATH"` 检查**非密钥**字段。

```bash
ROOT=/public/home/wwxia/yuang.li
AM=$ROOT/workspace/AgentMemory
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
cd "$AM"
bash -n .env
set -a
source .env
set +a
export PYTHONPATH="$AM:$AM/methods/RecMem"
test -f "$LOCOMO_PATH" && test -f "$LONGMEMEVAL_S_PATH"
test -d "$NLTK_DATA/tokenizers/punkt" && test -d "$NLTK_DATA/tokenizers/punkt_tab"
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8001/v1/models
```

Conda 环境分别为 `$ROOT/miniforge3/envs/recmem`、`recmem-memoryos`、`recmem-mem0`、`recmem-amem`；不要将四套依赖混装。正式复现实验应使用 `$ROOT/models/Qwen3-VL-32B-Instruct` 生成服务和 `$ROOT/models/Qwen3-Embedding-0.6B` embedding 服务；2026-10-02 的联调实际使用 Qwen3-4B，见下方记录，不能冒充 32B 结果。API 返回的服务模型名须与 `.env` 中的 `MODEL`、`EMBEDDING_MODEL` 一致。若两个服务在不同节点，检查端口转发/可达地址，不能默认 `127.0.0.1` 跨节点可用。资源受限时先单对话、低并发；不得占用他人的 GPU。

## 运行：先 smoke，再全量

以下命令在同一 Bash 终端执行。LoCoMo smoke 用 `--conv_limit 1`；LongMemEval-S smoke 将 `BENCH` 改为 `longmemeval_s`、`DATA` 改为 `$LONGMEMEVAL_S_PATH`、`SCORE_DIR` 改为 `$LONGMEMEVAL_S_SCORE_PATH`，选择参数改为 `SELECTION=(--sample_size 1)`。全量时用 `SCOPE=full` 与 `SELECTION=()`，务必使用**新的**答案/存储路径，不覆盖 smoke 结果。

```bash
BENCH=locomo
SCOPE=smoke
DATA=$LOCOMO_PATH
SCORE_DIR=$LOCOMO_SCORE_PATH
SELECTION=(--conv_limit 1)
RESULTS=$AM/artifacts/results/comparison/$BENCH/$SCOPE
STORE_ROOT=$AM/artifacts/stores/comparison/$BENCH/$SCOPE
PREFIX=${BENCH}_${SCOPE}
mkdir -p "$RESULTS" "$STORE_ROOT" "$SCORE_DIR"

conda activate "$ROOT/miniforge3/envs/recmem"
for METHOD in recmem fullcontext naive_rag; do
  EXTRA=()
  if [[ $METHOD == recmem ]]; then
    EXTRA=(--semantic_store "$STORE_ROOT/recmem/semantic"
           --subconscious_store "$STORE_ROOT/recmem/subconscious"
           --episodic_store "$STORE_ROOT/recmem/episodic")
  elif [[ $METHOD == naive_rag ]]; then
    EXTRA=(--rag_store "$STORE_ROOT/naive_rag" --top_k 10 --context_budget 28000)
  else
    EXTRA=(--context_budget 28000)
  fi
  python -m evaluation.run_experiments \
    --method "$METHOD" --bench "$BENCH" --model "$MODEL" --judge_model "$MODEL" \
    --output_file "$RESULTS/${PREFIX}_${METHOD}.json" \
    --conv_workers 1 --question_max_workers 1 \
    "${EXTRA[@]}" "${SELECTION[@]}" || break
done
```

前三种方法运行结束会自动评分。外部方法各用自己的环境；评分仍切回 RecMem 环境。外部存储根路径不能共用，Mem0 的数据库/遥测也必须留在个人工作区。不要在旧结果上直接用 `--overwrite`；中断后对**同一命令**加 `--resume`。`--resume` 与 `--enable_stat` 不能合用。

```bash
for METHOD in memoryos mem0 amem; do
  conda activate "$ROOT/miniforge3/envs/recmem-$METHOD" || break
  python -m evaluation.external_runner \
    --method "$METHOD" --bench "$BENCH" --data_path "$DATA" \
    --output_file "$RESULTS/${PREFIX}_${METHOD}.json" \
    --store_root "$STORE_ROOT/$METHOD" \
    --model "$MODEL" --embedding_model "$EMBEDDING_MODEL" \
    --embedding_dim "$EMBEDDING_DIM" --top_k 10 --context_budget 28000 \
    "${SELECTION[@]}" || break

  conda activate "$ROOT/miniforge3/envs/recmem" || break
  python -m evaluation.run_experiments --bench "$BENCH" --eval_only \
    --output_file "$RESULTS/${PREFIX}_${METHOD}.json" --judge_model "$MODEL" || break
done
```

六份评分文件齐备后汇总；不一致的题目、标准答案、类别、数据哈希、服务模型或评分来源会被拒绝。`SCORE_DIR` 必须与当前 bench 对应。

```bash
conda activate "$ROOT/miniforge3/envs/recmem"
RUNS=()
for METHOD in recmem fullcontext naive_rag memoryos mem0 amem; do
  RUNS+=(--run "$METHOD=$RESULTS/${PREFIX}_${METHOD}.json=$SCORE_DIR/${PREFIX}_${METHOD}.json")
done
python -m evaluation.compare_results \
  --bench "$BENCH" --data_path "$DATA" "${SELECTION[@]}" \
  "${RUNS[@]}" --output "$RESULTS/comparison.json"
```

## 历史结果迁移（已执行，供审计及新节点复现）

2026-10-02 已把本地**新代码结构**同步到远程 `$AM/methods/`，排除本地 `.env`、`artifacts/`、缓存和模型。以下是当时执行的审计式迁移命令；在另一节点重复时仍应先确认 `pwd`、所有目标路径都在个人目录，且新方法源码已同步完整。不要对已迁移的同一数据再次执行 `--apply`。

```bash
cd "$AM"
conda activate "$ROOT/miniforge3/envs/recmem"
export PERSONAL_ROOT="$ROOT"
python -m evaluation.migrate_layout --old-project "$AM/RecMem" --new-root "$AM"
# 只有预览通过、确认目标不冲突后才执行：
python -m evaluation.migrate_layout --old-project "$AM/RecMem" --new-root "$AM" --apply
```

默认仅预览：核对旧/新路径、答案与数据 SHA-256、评分来源哈希、清单字段及目标冲突；`--apply` 才搬运旧 `dataset/`、`results/`、`.cache/`、`logs/`、`.env` 到 `artifacts/`/根目录，并把 `results` 内的 `stores` 和 `service-logs` 分流。脚本在 `artifacts/migration/backup/` 保存原始清单；只重写清单中的绝对路径和评分清单的 `answer_file`，不改答案/评分 JSON 内容，也不删除旧源码。每个答案旁的 `.migration.json` 记录旧/新源码哈希、清单哈希、数据哈希和迁移前已完成的对话编号。**旧清单仍保留旧源码哈希**；迁移后新增对话属于记录中的新源码版本，不能与旧结果伪称同版。正常代码/参数/数据变更仍禁止续跑，只有具有匹配迁移记录与备份的旧源码哈希可获例外。远程迁移前另存一份可恢复备份；若目标已有文件或存在未知清单字段，停止并人工核对，不要用覆盖开关绕过。

## 离线验收与常见故障

本地 PowerShell 测试时，在工作区根目录设置 `PYTHONPATH` 为 `AgentMemory;AgentMemory\methods\RecMem` 两个绝对路径，然后在 `methods\RecMem` 运行 `python -m unittest discover -s tests -q`。远程 Linux 用冒号分隔。三个入口用 `python -m evaluation.run_experiments --help`、`python -m evaluation.external_runner --help`、`python -m evaluation.compare_results --help` 检查。迁移工具另有 `--help`。

- `ModuleNotFoundError: methods`：`PYTHONPATH` 少了 AgentMemory 根路径。
- `FileExistsError`：目标结果/存储已存在；优先 `--resume`，若参数或源码不匹配，新建独立结果路径，不要清空公共目录。
- `Cannot resume`：核对清单、数据哈希、源码哈希与迁移记录；手改清单会破坏审计。
- NLTK 联网报错：检查 `NLTK_DATA/tokenizers/{punkt,punkt_tab}`；GPU 节点不一定可访问公网。
- vLLM 400 上下文超限：分别检查服务的 `max_model_len`、方法内部 extraction 的 `max_tokens` 与回答上下文预算；单纯调大输出 token 可能更易超限。
- 汇总拒绝：逐份检查答案、评分 `*.run.json`、题目抽样参数、`judge_model` 与 `answer_sha256`，不要直接拼接不同实验。

旧版结果没有统一的逐请求 token/API 调用次数；现有评分只提供 LLM judge、BLEU 和 F1 等指标。下述新计量运行与旧结果分开保存。

## 2026-10-02 远程同步与六方法联调记录

- 新代码在 `$AM/methods/`，历史数据、答案、评分、日志、存储和 NLTK 缓存在 `$AM/artifacts/`；原运行清单及旧版备份在 `$AM/artifacts/migration/`。从旧独立 RecMem 项目迁入的结果与 store 分别在 `artifacts/results/legacy-RecMem/`、`artifacts/stores/legacy-RecMem/`，日志在 `artifacts/logs/legacy-RecMem/`。原答案/评分 JSON 未改写；这批更早的结果没有 `*.run.json`，不能直接用于当前严格汇总或断点续跑。
- 联调数据为真实 LoCoMo 第一个对话的第一个 session（18 条消息）和前两道题，文件为 `$AM/artifacts/data/smoke/locomo_one_session_two_questions.json`，SHA-256 为 `cc25b8381b2981b78cd83d05f017cf808fb95c9596c774cc3d964d175c5451d1`。这只是接口、评测和目录迁移检查，不能当作正式 LoCoMo 分数。
- 六份答案及汇总在 `$AM/artifacts/results/comparison/locomo/reorg-mini/`，对应隔离存储在 `$AM/artifacts/stores/comparison/locomo/reorg-mini/`，日志在 `$AM/artifacts/logs/reorg-mini/`。`comparison.json` 的六方法逐题/哈希/评分来源核验已通过；每种方法 2 题，LLM judge 分数依次为 RecMem 0.5、FullContext 1.0、Naive RAG 0.5、MemoryOS 0.5、Mem0 0.0、A-Mem 0.5。分数的样本量太小，不能用于方法优劣结论。
- 联调时 8000 端口为 Qwen3-4B-Instruct-2507 的 `generate` 服务，服务名 `qwen3-4b-chat`、上下文 16384；8001 端口为现有 Qwen3-Embedding-0.6B。先前 8000 端口误以 `pooling` 模式启动，无法提供 Chat Completions，已只重启本人该端口的服务。正式实验需先确认所需的 32B 生成服务、显存和模型名，再使用**全新的**答案及 store 路径。
- 远程 `recmem` 环境在清理后离线测试为 22 项通过。旧独立 `$ROOT/workspace/RecMem` 和旧 `$AM/{RecMem,MemoryOS,mem0,A-mem,.cache}` 已在核对/归档后删除；旧源码的可恢复归档保存在 `$AM/artifacts/migration/archives/`。**保留 `$AM` 根目录**，它正是新工作区。清理后两个模型端点仍正常响应。

## LoCoMo 全量 4B 流程验证（已完成）

2026-10-02 使用完整的 10 个 LoCoMo 对话（数据 SHA-256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`）运行六种方法。用户明确选择沿用 `qwen3-4b-chat`，embedding 为 `qwen3-embedding-0.6b`，回答上下文估算预算 12,000 tokens，评分也使用同一 4B 端点。这是**完整数据集上的 4B 流程验证**，不是 Qwen3-VL-32B 论文配置结果。

答案和评分在 `$AM/artifacts/results/comparison/locomo/full-4b-20261002/`，隔离存储在 `$AM/artifacts/stores/comparison/locomo/full-4b-20261002/`，日志在 `$AM/artifacts/logs/full-4b-20261002/`。六方法的 10 个对话已完成；Mem0 中断后经审计式源码迁移使用 `--resume` 续跑，`driver-resume.log` 于 2026-10-03 14:25:21 记录严格汇总完成。旧答案和评分不参与下述新计量目录。

## 六方法 token 与耗时计量（4B 独立重测）

先在已有 GPU 作业的 `gpu10` shell 中执行 `bash "$AM/run_locomo_measure4b.sh" --smoke`，核对单对话两题的计量与评分；通过后执行 `bash "$AM/run_locomo_measure4b.sh"` 跑完整 LoCoMo。不要在登录节点直接运行。脚本先验证数据 SHA-256 和 8000/8001 两个模型名，再顺序运行六种方法。完整重测的新答案/评分、store、日志分别位于 `artifacts/results/comparison/locomo/full-4b-measured-20261003/`、`artifacts/stores/comparison/locomo/full-4b-measured-20261003/`、`artifacts/logs/full-4b-measured-20261003/`；smoke 路径中的目录名改为 `smoke-4b-measured-20261003`。默认 `--resume --measure`，但任何源码、数据或参数不匹配仍会拒绝续跑；不要使用 `--overwrite` 混合版本。计量运行统一 `conv_workers=1`、`question_max_workers=1`，不同于旧版并发配置。

每份答案旁的 `.metrics.json` 按对话保存构建耗时、每题查询/检索/回答耗时、LLM 输入/输出 token、embedding 输入 token、调用和失败次数；judge 耗时单列。`compare_results --metrics` 只有在逐题、数据哈希、答案评分来源和计量完整性都核对通过后才写入汇总的 `efficiency` 部分。Construct Token 为构建阶段 LLM 输入+输出 token 除以对话数，Query Token 为检索+回答阶段 LLM 输入+输出 token 除以题数；embedding 独立列示，缺失服务 `usage` 时对应均值为 `null`，不能当作 0。外部 SDK 的隐式 HTTP 重试无法从成功响应准确还原；`failed` 只统计在 Python 调用边界可见的失败。A-Mem 是内存态 Chroma 适配，磁盘目录大小不能与其他方法直接比较。

2026-10-03 的 smoke 已完成六方法各 1 个对话、2 道题，`comparison.json` 的逐题、哈希和计量完整性校验通过，所有实际调用的 LLM/embedding `usage` 均有返回。完整数据计量重测已在同一 gpu10 Slurm 作业中启动，进度看 `artifacts/logs/full-4b-measured-20261003/driver.log`；不要重复启动。
