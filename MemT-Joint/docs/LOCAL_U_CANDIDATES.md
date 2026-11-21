# 短 episode 的 Local-U 候选

Local-U 从同一个 Builder 后状态出发，固定同一批 fact/experience 候选，采样多个 Updater 决策；在相同后续对话和问题上与不执行当前 U 的基线比较。奖励继续使用问答收益差、记忆成本和编辑成本，组内标准化及训练权重不变。

“写入 QA”指证据涉及当前 block 的问题，“保留 QA”指全部证据来自之前已观察 block 的问题。它们不要求已经答对，也不保证已经存入可检索记忆；QA 和标准答案不进入 Builder/Updater 提示词。锚点至少有这两类问题，尽量各选一半，并补足原有 local_questions 配额。之后回放剩余 block，统一在 episode 末评估。

253988 的磁盘故障与本问题不同。254342 第8步失败时，符合双 QA 的 block 3 在路径2/3上分别有4/3个合法候选，block 6 在路径3上有7个；原代码仅检查路径0，因此误报没有候选。编号均从0开始。

修复保留每个 block 优先路径0的行为；路径0无候选时按顺序检查已有其他全局分支，使用该分支自己的 Builder 前后状态。锚点选择不使用奖励或答题分数。仅当所有合格 block 的全部全局分支都无候选时，随机选择一个双 QA 锚点，追加一批当前策略 Builder 采样，批大小沿用 builder_rollouts，最多追加这一批。重采样输出只是锚点上下文；原全局输出和违规惩罚继续保留，不伪造候选。

按当前实验要求，QA 覆盖不足或 Builder 连续无合法 fact/experience 候选时跳过 Local-U，并分别记录 `U.local: no_qa_coverage` 或 `U.local: no_legal_candidates`。如果全局路径也没有任何待更新候选，记录 `U.global: no_legal_candidates`。正常情况下不增加模型调用。

跳过 Local-U 时，Local-B 在最后一个 block 的真实 Builder 前状态重新采样，并使用已审校的终局 QA 评分；Global-B、已有 Global-U 样本和 R 的 MoT 继续训练。省去无用的 before-U 问答比较，不伪造 Updater 样本，不把其他角色的样本复制成 U。

缺失分支的实际权重为0，其余分支维持原权重，不重新归一或补给其他角色。因此，仅跳过 Local-U 的step有效总权重为5/6；Global-U、Local-U均跳过时为2/3。`metrics.json`逐步记录 `skipped_branches`、`active_weight_mass`、各分支/角色权重和样本数；完整轨迹会反映实际跳过比例。训练和速度验收允许有记录的U分支缺失，仍拒绝未记录的分支丢失或权重变化。

双 QA 仍用于可用 Local-U 锚点的信用分配，不是所有记忆写入必须满足的执行规则。数据库、预算、推理后端或优化器异常仍正常抛出，不归入“无候选”跳过。本次实验继续执行其余可用分支，不能将所有50步都描述为完整B/U/R联合更新。
