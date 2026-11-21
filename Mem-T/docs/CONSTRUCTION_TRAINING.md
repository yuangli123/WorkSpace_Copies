# 构建训练：运行、验收与恢复

## 方法与实现假设

依据论文§3.1、§3.3 Eq.16–18及A.2：先采样构建动作，用冻结检索策略得到森林；对叶子优势乘来源证据gate与0.1倍trace gate，按问题内叶子、再按episode问题平均；按阶段及具体工具类别保留前50%，进行离线最大似然SFT。构建侧不额外使用GRPO更新。

`legacy-v1`沿用公开代码的原始对话自动归档。`paper-v2`新增`create_raw(turn_ids)`，仅允许当前输入块的唯一真实ID，按输入顺序复制原文；工作记忆与原始归档直接写入，事实/经验经过ADD/UPDATE/DELETE/IGNORE。CrtRaw工具细节未公开，这是明确记录的研究设计。

UPDATE继承来源与生产链；DELETE/IGNORE没有新输出时只可能获得evidence credit，不制造trace命中。该血缘解释不是因果证明。当前实现按具体工具类别排名；persona和summary均作为工作记忆子类型。无效completion中的有效片段也不得进入训练集。

主SFT使用Trainer+DeepSpeed ZeRO-3，而非论文的LLaMA-Factory框架，以精确保留任意原子动作mask。LF整completion导出仍可用，但会排除部分入选的completion，不能声称与原子mask等价。构建与检索各从base独立初始化是本研究的角色约定，非作者已公开的最终权重衔接说明。

## 短程配置

`configs/experiments/construction-smoke4.yaml`：4卡、每卡batch2、累积4、全局batch32、4步、BF16/FA2、最大6144 token、AdamW、cosine、5e-6、warmup0.1、seed42。90/10按原始completion分组，同源副本中完全相同的prompt/response也放在同一侧。为使每次更新严格使用完整batch，训练epoch尾部不足32条的部分不进入该epoch的数据表，并记录排除数量；不把它记为额外的10k采样数据。

默认采训练集8题森林；若无非零、非退化且有trace的归因，则在同一记忆快照上扩至全部152题。两种构建模型均处理完整验证对话，分别新建记忆，用固定的验证8题和固定检索器比较。

真实4B仅最终HF导出，不保存大型优化器检查点。独立微型Qwen3通过同一4卡ZeRO-3路径连续训练4步，从完整step2恢复再到step4，比较参数、优化器/调度器状态及四rank RNG文件。不得将此结果写成真实4B完整续训已验收。

微型恢复对照显式启用确定性FlashAttention、cuBLAS和固定NCCL算法；在更新前逐rank核对FP32主权重、Adam状态、BF16分片、调度器、CPU/CUDA RNG及实际microbatch哈希。真实4B短程仍使用正常训练设置，不把默认非确定性内核造成的量化差异当成恢复验收通过。

实卡审计进一步发现CPUAdam的C++ β幂缓存不在state_dict中：连续步使用连乘，恢复后跳步使用pow，能在BF16边界产生差异，即使主权重、矩、步数、RNG和数据均完全恢复。项目在每个实际CPUAdam step前重建仅含标量缓存的内核对象，保留Python矩、主权重、计数和scheduler hook，使两条路径使用一致的幂计算；不修改安装的DeepSpeed库。训练审计记录该修正是否生效。

```bash
# 在个人memt-train环境和Mem-T项目目录内
python -m cli run-construction --config configs/experiments/construction-smoke4.yaml --preflight
# 通过调度器申请自己的4张GPU；不在登录节点执行训练
sbatch scripts/slurm_train.sh
# 重新提交同一流程时必须核对无存活的相同作业
python -m cli run-construction --config configs/experiments/construction-smoke4.yaml --resume
```

预检查也建立实验身份记录，因此之后使用`--resume`。查看`CURRENT.json`须同时核对Slurm及实际进程；文件或锁的存在不证明正在运行。观察超时不重启。

## 状态与失败

- `stages/*.json`记录完整命令、输入SHA和输出SHA；完整阶段校验失败时拒绝跳过。
- `progress/<sample>/CURRENT.json`只指向封存完整输入块。恢复到新的独立数据库，避免沿用中断时未提交的副作用。
- 森林逐QA落盘，恢复核对QA、顺序、种子及源码/配置身份。
- export_only SFT没有中途优化器恢复能力；失败阶段归档在本run内，再由显式`--resume`从base重做，预算仍包含归档内容。
- 非有限loss/梯度、空监督token、归因全零、输入漂移、超预算或不完整导出均失败，不发布成功标记。

## 预算

沿用已有十进制100GB口径：项目、基础模型、BGE及既有共享缓存审计范围；既有Conda环境单列排除。90GB预警，100GB硬上限，保留2GB运行余量。短程按HF导出峰值准入，正式full模式按优化器和双检查点轮换的保守峰值准入。预算不足时停止，不删除历史结果，不偷偷改为低精度优化器或LoRA。

`construction-200.yaml`是未自动启动的正式配置：200步、完整状态保存、初始42次独立构建rollout、至少10000个真实有效入选操作。数量不足则在训练前报错，需在新run中增加独立rollout；不复制操作凑数，不宣称与作者未公开10k数据一致。当前100GB剩余空间通常无法容纳该full配置，须先解决预算，再执行。

## 验收产物

`COMPLETE.json`只有在所有阶段通过后生成，附带阶段receipt、报告和对比分数哈希。报告包括真实参数变化、4B重载执行工具、全局batch/步数、微型四卡完整续训证据、新协议同条件F1/BLEU-1/judge和磁盘审计。

短程分数仅用于流程验收，不要求4步后提升；10k数据采样、正式200步构建训练、1307题最终复测和多种子消融不属于此次短程执行范围。

## 根目录迁移后

入口统一为`python cli.py`或`python -m cli`。短程配置现输出到独立的`artifacts/construction-smoke4-root-layout`；旧完成结果仍位于`artifacts/construction-smoke4-20261102`。历史run的源码身份保持冻结，新代码不会绕过校验原地续跑。此前同步的小型验收记录现位于`artifacts/validation/`。
