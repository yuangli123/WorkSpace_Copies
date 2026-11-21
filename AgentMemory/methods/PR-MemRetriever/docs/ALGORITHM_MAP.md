# 论文到代码

| 论文 | 实现 |
| --- | --- |
| §3.1 memory formation/evolution | `methods.construction.rollout.construct_episode` 复用 `memory/formation_tools.py` 和 `memory/update.py` 的提示词和工具 |
| §3.1 raw/factual/experiential/working memory | 上游工具接口；`memory.journal.JournalDB` 保存所有版本与操作血缘 |
| Eq.8-9 多棵树、pivot续写 | `methods.retrieval.tree.sample_tree`, `methods.retrieval.tree.sample_forest` |
| 多卡批量采样 | `methods.retrieval.tree.sample_forests` 将不同问题/分支的活动节点成批发送，每条请求独立seed |
| Eq.10 format/evidence/performance | `methods.retrieval.tree.score_forest`；子节点均值递归 |
| Eq.11-13 intra/inter advantages | `methods.retrieval.tree.standardize`, `methods.retrieval.tree.score_forest` |
| Eq.14-15 clipping/KL | `methods.retrieval.loss.mot_loss`；native `HFUpdater` 与分布式 `MoTActor` 共用 |
| Eq.16-17 hindsight score | `methods.construction.credit.hindsight_scores` |
| Eq.18 按类筛选并最大似然训练 | `methods.construction.credit.select_operations`, `training.sft.sft_example`, `training.sft.train_sft` |
| A.2 分布式训练基础 | `training.verl_backend.VerlBackend` 复用固定版本 Ray/vLLM/FSDP worker；不使用其原版 outcome-only loss |
| 回溯数据采集 | `experiments.workflow.collect_forests` 固定最终检索策略重新采样训练问题 |
| 闭环与复现入口 | `cli`，`experiments/construction.py` |

上游改动涉及：延迟导入 vLLM/Chroma，使纯逻辑和CPU测试可运行；本地 embedding 路径；修复 Chroma 读接口插入 dummy、query 参数/默认字段与错误吞没；给新训练入口提供确定性 ADD ID；移除 main.py 对不存在函数的导入。Tree-GRPO 中添加 actor 选择及 per-request seed 传递。正式训练入口为新 CLI，旧 main.py 已归档，活动源码只保留根目录 CLI。

基础模型上下文参考：[Qwen3-4B config](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)。可选 LLaMA-Factory 导出参考：[v0.9.3 Qwen3 template](https://github.com/hiyouga/LLaMA-Factory/blob/v0.9.3/src/llamafactory/data/template.py)，采用 `template=qwen3, enable_thinking=false`。
