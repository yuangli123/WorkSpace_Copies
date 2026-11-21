# MemT-Joint

在Mem-T四类持久记忆上实现共享联合RL：Builder/Updater使用Global＋Local，Retriever使用固定库MoT，三者共享参数。新版短窗口 M4 的 Answer 使用当前共享权重，仅推理、不增加 Answer loss；旧配置保留冻结 Answer。目录分层与工具代码参考本地Mem-T，项目不依赖原目录运行。

主线保留paper-v2的`create_raw`，不加入持久语义树、Triggers或主方法SFT。两阶段SFT只用于M6对照。

当前执行[短窗口 v2](docs/WINDOW_V2.md)：8块窗口、审校的窗口末QA、驻留和批量生成、共享Answer，先验收3次真实更新的步时。它沿用40K预算并重算完整81题基线。[旧统一协议](docs/COMMON_PROTOCOL.md)及下方命令保留作历史流程参考。

```text
MemT-Joint/
├── cli.py                  # 数据、采样、训练、恢复验收、导出、评测、judge、比较
├── models/                 # HF、固定模型、共享运行时与证据队列
├── memory/                 # Mem-T工具、四类分支状态、版本和BGE排序
├── methods/joint/          # Global/Local、并行MoT、信用分配、step-ratio损失
├── training/               # HF共享更新、FSDP/vLLM、M5独立参数、M6原子SFT
├── experiments/            # 配置、编排、验证选模、恢复、运行身份
├── evaluation/             # 共同证据协议、judge、对话聚类比较
├── configs/                # 正式、profile、HF、fixture、M5/M6
├── scripts/                # CPU检查、合成数据、四卡Slurm入口
├── tests/                  # 环境、数值、真实CPU微型更新和恢复测试
├── third_party/            # 固定版本Tree-GRPO及许可证
├── artifacts/              # 本项目运行产物，默认不入Git
└── docs/                   # 已确认设计与实现边界
```

## 本地先检查

从项目根目录执行；检查脚本禁用GPU，不下载模型、不安装或修改系统环境。

```powershell
Set-Location 'D:\Desktop\工作文件\MemT-Joint'
python -m scripts.check
python -m scripts.make_fixture --output artifacts/demo-data
python cli.py smoke --config configs/fixture.json --data artifacts/demo-data/train --output artifacts/demo-smoke
python cli.py evaluate --config configs/fixture.json --data artifacts/demo-data/test --output artifacts/demo-eval
```

已有目录不覆盖；重复运行请选择新输出名。`smoke`只检查合成环境，`SMOKE_COMPLETE.json`明确记录零优化器更新；`train --config configs/fixture.json`被拒绝。测试中的微型Torch策略真实采样、求梯度、更新与恢复，但其离散工具词表不是Qwen模型，也不是数据集成绩。

实际检查结果写入`artifacts/validation/LOCAL_CHECKS.json`。Windows现有Anaconda的pyreadline可能在退出时打印外部兼容警告；检查脚本预先加载Torch并隔离pytest插件，未修改这些安装包。完整训练环境要求Linux/Python 3.11，依赖见`requirements.txt`；CPU测试依赖单列`requirements-test.txt`。

## 远程准备与主实验

以下是待执行命令，本次没有同步、申请GPU或启动远程训练。代码应位于个人`workspace/MemT-Joint`，训练产物使用个人AgentMemory的独立run路径。

```bash
ROOT=/public/home/wwxia/yuang.li
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
# 若该新环境不存在，可克隆本人已经可用的环境；已有环境不要覆盖。
test ! -e "$ROOT/miniforge3/envs/memt-joint"
conda create -y -p "$ROOT/miniforge3/envs/memt-joint" \
  --clone "$ROOT/miniforge3/envs/memt-train"
conda activate "$ROOT/miniforge3/envs/memt-joint"
python -m pip check
python -c 'import torch, transformers, ray, vllm, flash_attn, deepspeed, sentence_transformers; print(torch.__version__, transformers.__version__)'
cd "$ROOT/workspace/MemT-Joint"
export PYTHONPATH="$PWD:$PWD/third_party/Tree-GRPO"
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full"
test -f "$DATA/train/episodes.jsonl"
test -f "$DATA/valid/questions.jsonl"
python cli.py preflight --config configs/joint.json --data "$DATA/train"
```

`DATA`是现有复现配置记录的位置，使用前仍须检查真实文件。也可通过`prepare --source <原始LoCoMo JSON> --output <新目录>`生成1/1/8划分；LongMemEval用`prepare --benchmark longmemeval`，只作OOD评测。

先运行10轮profile，逐轮保留恢复点；正式实验使用新run和200次共享更新，不自动把profile权重当作正式初始权重。

```bash
RUNS="$ROOT/workspace/AgentMemory/artifacts/memt-joint"
sbatch scripts/slurm_train.sh train --config configs/profile.json \
  --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUNS/profile-seed42"
sbatch scripts/slurm_train.sh train --config configs/joint.json --variant M4 \
  --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUNS/M4-seed42"
```

第二条命令应在profile验收和规模确认后执行，不同时提交重复实验。修改`seed`或任何采样参数请生成独立配置与run，再跑其他种子。正式R Local目标每更新32个问题组，期间策略锁定；采样按活动节点成批发送，显存microbatch与问题组数量分开。

默认完整历史训练，因为原生LoCoMo QA是终态标注；`joint.curriculum`可配置前缀课程，但短前缀须有`--probes`提供的人工核验题，否则明确失败，不臆造在线标签。探针需`reviewed=true`、零起始`eval_session`，证据不得来自之后的会话。

早期Local若没有当前合法题，默认以冻结旧策略顺序续跑未来对话，到终态再评估候选。候选自身看不到未来；续跑调用不接梯度、单独计入成本。`local_evaluation=current_only`可作无续跑诊断，但原生终态QA下会跳过大量早期anchor。

## 恢复、选模和导出

每次运行绑定代码、配置、数据、基础权重、Tokenizer、embedding和环境身份。恢复始终使用最新完整checkpoint；测试集不能用于选模。

```bash
sbatch scripts/slurm_train.sh train --config configs/joint.json --variant M4 \
  --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUNS/M4-seed42" --resume
sbatch scripts/slurm_train.sh verify-restore --run "$RUNS/M4-seed42" \
  --output "$RUNS/M4-seed42-restore-probe"
python cli.py export --run "$RUNS/M4-seed42" --selection best \
  --output "$RUNS/M4-seed42-hf"
```

`verify-restore`实际加载、重新保存并逐Tensor比较受信任检查点。它不凭文件存在宣称恢复，也不承诺GPU继续训练的逐位等价。验证集评测会用当前B/U重新建库；`BEST.json`根据valid F1选择，`LATEST.json`用于续训。未提供`--valid-data`的训练标记为`final_only_no_validation`，不宣称完成验证选模。

在run目录创建名为`STOP`的文件会请求更新边界保存后退出；恢复前由操作者移除自己的STOP文件。Slurm默认六小时与提前180秒信号仅为入口模板，不能保证长轮次在180秒内结束。根据profile调整可申请时限或更早请求停止；超时只认最近完整恢复点。正式默认每5轮保存，profile每轮保存。

## 完整评测与原文诊断

```bash
python cli.py evaluate --config configs/hf-main.json --variant M4 \
  --model "$RUNS/M4-seed42-hf" --data "$DATA/test" --output "$RUNS/M4-test"
python cli.py evaluate --config configs/hf-main.json --variant M4 \
  --model "$RUNS/M4-seed42-hf" --mode memory_only \
  --data "$DATA/test" --output "$RUNS/M4-test-memory-only"
python cli.py judge --config configs/hf-main.json --answers "$RUNS/M4-test/answers.jsonl" \
  --protocol memory-r2-generous-v1 --output "$RUNS/M4-test-judge"
```

Memory-only重新运行检索，隐藏来源原文并禁用`search_turns`。Source-backed只计真正显示且提交的来源ID，不能将元数据中的全部来源视为命中。Raw正文通过`sources`显示一次；所有模型输出、工具输出和固定Answer调用均计账。

LongMemEval对其prepared目录执行相同evaluate；再运行`judge --protocol longmemeval-official-prompt`。该协议采用官方题型prompt与严格yes/no解析，必须报告实际judge模型；不能把Qwen判断当成GPT-4o结果，也不把F1改名为Accuracy。没有完整judge覆盖时不发布完整Accuracy。

## 对照与消融

| 组 | 命令选择 | 实际含义 |
|---|---|---|
| M0 | evaluate --variant M0，不给训练模型 | 冻结基础模型基线 |
| M1 | train --variant M1 | B/U确实走固定基础权重，只有R更新 |
| M2 | train --variant M2 | R确实走固定基础权重，只有B/U更新 |
| M3 | train --variant M3 | B/U仅Global，R仍MoT |
| M4 | train --variant M4 | 共享B/U Global+Local、R MoT |
| M5 | configs/separate.json，--variant M5 | 三套独立参数与优化器，同生命周期目标 |
| M6 | 下述两阶段命令 | R训练、回溯筛选、构建原子SFT |

M5使用HF分时卸载实现，不同时常驻三套模型。**共享收益应配对比较M4-HF与M5-HF**（`hf-main.json`与`separate.json`），不要把M4的FSDP后端与M5的HF后端差异误归因为参数共享。独立模型可能需要更大主机内存；四卡FSDP主线及M5真实资源峰值仍待实测。

M6按顺序运行，每一阶段均用新目录；第一个阶段就是M1：

```bash
# 完成M1训练后，先export其R权重。
python cli.py collect-sft --config configs/two-stage.json --data "$DATA/train" \
  --retrieval-model "$RUNS/M1-hf" --output "$RUNS/M6-supervision"
sbatch scripts/slurm_train.sh sft --config configs/two-stage.json \
  --selected "$RUNS/M6-supervision/selected.jsonl" --output "$RUNS/M6-construction"
python cli.py evaluate --config configs/hf-main.json --data "$DATA/test" \
  --construction-model "$RUNS/M6-construction/final" --retrieval-model "$RUNS/M1-hf" \
  --output "$RUNS/M6-test"
```

采集器采用固定训练后R、真实独立构建rollout、Mem-T回溯双gate、按类前50%和原子span。约10k实际入选操作后才进入200步SFT；数量不足或归因全零直接失败，不复制数据凑数。这是共同接口下的两阶段训练范式对照，不冒称原论文数字。

配置中的`joint.loss_variant=token`可比较逐token裁剪；`step`是主方法。奖励系数有范围校验，修改需新run。额外R Global损失不在主方法中，不能通过改个标签声称已实现该消融。

```bash
python cli.py compare --answers M1="$RUNS/M1-test/answers.jsonl" M4="$RUNS/M4-test/answers.jsonl" \
  --output "$RUNS/comparison.json"
```

比较器严格核对QA覆盖、题目和gold，按对话聚类bootstrap。训练时间请使用`telemetry/*/attempt.json`，包含初始化、保存、验证；仅看`metrics.json`的采样/优化时间会漏掉开销。

## 当前验证边界

已提供CPU机制与真实微型更新测试、主流程、M0–M6对照入口、周期验证、完整恢复验收命令、导出和评测。**没有运行真实Qwen3-4B、四卡FSDP、M6四卡SFT或正式benchmark；不能据CPU测试宣称这些已实卡验收。**

来源、存储后端适配和实现差异见[IMPLEMENTATION.md](docs/IMPLEMENTATION.md)；已确认研究设计保存在[DESIGN.md](docs/DESIGN.md)。
