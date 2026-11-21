# Mem-T训练机制研究

基于论文和公开推理工具的研究重实现。当前代码只保留一套根目录分层，统一入口为`python cli.py`或`python -m cli`。

```text
Mem-T/
├── cli.py          # 数据准备、训练、构建闭环和评测入口
├── models/         # HF/vLLM适配、Tokenizer、五个模型角色；不是权重目录
├── memory/         # 形成/演化/检索工具、CrtRaw、存储与来源血缘
├── methods/        # retrieval：MoT-GRPO；construction：采样与回溯归因
├── training/       # 检索更新、原子动作SFT、检查点和CPUAdam适配
├── evaluation/     # F1、BLEU-1、judge及HF导出重载验证
├── experiments/    # 流水线、阶段恢复、预算与运行状态
├── configs/        # models / methods / experiments
├── scripts/        # 四卡Slurm入口、缓存审计及诊断/报告工具
├── tests/          # 机制与迁移回归，含原版数值基准fixture
├── third_party/    # 固定版本Tree-GRPO及许可证
├── artifacts/      # 既有数据、模型、实验结果和历史源码快照
└── docs/           # 当前架构、方法、实验和迁移说明
```

- [架构与模型角色](docs/ARCHITECTURE.md)
- [训练和运行命令](TRAINING.md)
- [构建训练机制](docs/CONSTRUCTION_TRAINING.md)
- [实验结果索引](docs/EXPERIMENT_INDEX.md)
- [目录迁移与删除范围](docs/MIGRATION.md)
- [论文公式映射](docs/ALGORITHM_MAP.md)与[实现假设](docs/ASSUMPTIONS.md)

构建闭环：采样构建动作 → 冻结检索器采森林 → 双gate归因 → 按类筛选 → 原子mask SFT → 重新建库 → 同条件评测。

既有200步检索与4步构建权重保留在`artifacts/`。原始Qwen3-4B和BGE权重仍位于个人远程`/public/home/wwxia/yuang.li/models/`，不迁入代码目录。旧报告移至`artifacts/validation/`，旧源码只在`artifacts/code-snapshots/`封存。

新布局下的短程配置使用独立输出`artifacts/construction-smoke4-root-layout`；历史run不重写、不原地续跑。本次目录整理不启动训练。
