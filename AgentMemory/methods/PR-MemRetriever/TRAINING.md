# 运行与训练

## 环境与入口

```bash
cd /public/home/wwxia/yuang.li/workspace/Mem-T
source /public/home/wwxia/yuang.li/miniforge3/etc/profile.d/conda.sh
conda activate memt-train
python cli.py --help
```

复用现有个人环境，本次迁移不安装或升级依赖。`requirements.txt`是此前已使用的训练环境依赖；FlashAttention单独安装的说明见[环境记录](docs/REMOTE_SETUP.md)。

## 四卡构建闭环

```bash
# artifacts/已存在，Slurm日志写入该目录。此命令会申请并使用4张GPU。
sbatch scripts/slurm_train.sh
```

无额外参数时，脚本运行`python -m cli run-construction --config configs/experiments/construction-smoke4.yaml --resume`。新布局配置输出到`artifacts/construction-smoke4-root-layout`，不复用旧run。

流水线包含微型四卡完整恢复检查、构建动作采样、冻结检索器森林、归因筛选、真实4B全局batch32的4步SFT、HF导出/重载、两组完整验证对话重新建库、固定8题评测。完整完成才生成`COMPLETE.json`。

已有源码身份不同的实验不能直接使用新代码`--resume`；阶段、配置及输入校验仍生效。历史复核使用对应冻结源码。

## 独立阶段

统一子命令：`prepare`、`prepare-ood`、`build-memory`、`train-retrieval`、`collect-trees`、`derive-sft`、`train-sft`、`evaluate`、`export-llamafactory`。每个命令的参数见`python cli.py <command> --help`。

例如在自己申请的四卡上提交检索训练，所有输出必须选择新run：

```bash
sbatch scripts/slurm_train.sh train-retrieval   --config configs/memt4b_4gpu_100gb.yaml   --memory artifacts/<new-run>/memory-train   --valid-memory artifacts/<new-run>/memory-valid   --model /public/home/wwxia/yuang.li/models/Qwen3-4B   --backend verl --output artifacts/<new-run>/retrieval
```

构建SFT主后端为Trainer＋DeepSpeed ZeRO-3，形成与演化共用构建权重。四组件配置位于`configs/experiments/`，只用于明确选择construction/retrieval模型角色；不会自动全部执行。

## 数据与预算

实际已准备数据是`artifacts/retrieval-trial-20261026/data/full/{train,valid,test}`。已删除的根目录`data/`仅为占位README，不是实验数据。每次新记忆与结果都写入独立`artifacts/<run>/`。

100GB为十进制总预算；90GB预警，既有个人Conda环境单列排除。短程4B只导出最终HF权重。真实4B完整优化器续训未验证；正式200步/约10k操作/1307题配置须在实际峰值预算准入后运行，不能据微型恢复测试宣称真实4B完整续训已经验收。

## 回归检查

```bash
# 登录节点只执行CPU验证，不启动真实4B训练
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests
python -m evaluation.judge --self-test
bash -n scripts/slurm_train.sh
```

旧实验结果及其原始命令、代码哈希保持封存。新目录对应关系见[迁移说明](docs/MIGRATION.md)。
