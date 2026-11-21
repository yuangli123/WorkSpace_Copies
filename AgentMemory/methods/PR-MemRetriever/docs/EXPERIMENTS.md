# 训练机制研究

先固定 base 模型、数据哈希、对话划分、embedding、提示词、推理步数和评分。每次训练使用独立 output，不复用其他方法训练后的库。把“重新训练消融”和“评测时替换某个角色”的结果分开命名。

| 研究问题 | 变更（CLI --set） | 需要重新执行 |
| --- | --- | --- |
| 无训练基线 | 不运行任何训练，构建和检索均用 base | build train/valid/test、evaluate |
| 普通 GRPO | `retrieval.algorithm=grpo` | RL、最终森林、归因、SFT、测试库、评测 |
| 去除证据奖励 | `retrieval.evidence_alpha=0` | RL及后续全链路 |
| 去除 intra | `retrieval.use_intra=false` | RL及后续全链路 |
| 去除 inter | `retrieval.use_inter=false` | RL及后续全链路 |
| 仅训练检索 | 训练RL，测试库仍由base构建 | RL、base测试库、评测 |
| 仅训练构建 | 用base采最终森林并归因，SFT后检索仍用base | 最终森林、归因、SFT、测试库、评测 |
| 去除来源 evidence gate | `construction.evidence_gate=false` | 归因、SFT、测试库、评测 |
| 去除 retrieval trace gate | `construction.trace_gate=false` | 归因、SFT、测试库、评测 |
| 扩展轮数敏感性 | `retrieval.expansion_rounds=1/2` 分别跑 | RL及后续全链路 |
| 证据权重敏感性 | `retrieval.evidence_alpha=0/0.5/1` 分别跑 | RL及后续全链路 |

普通 GRPO 使用相同完整 rollout 次数上界，不等于相同 token 数。报告每个方法真实生成 token、树节点数量、训练 wall time、最终模型大小；不能仅按200次更新声称预算公平。

建议先进行 2 step / batch1 冒烟：`--set retrieval.steps=2 --set retrieval.batch_size=1 --set retrieval.save_every=1`。构建微调可用 `--set construction.steps=2 --set construction.save_every=1`；小数据仍应有两个不同 completion，以便独立验证。

机制证据应包括：奖励是否非零、优势分布是否退化、参数是否变化、KL是否有限、来源 gate/trace gate 的命中数量、按类筛选比例和并列分数，以及构建训练后重新生成的记忆是否改变。最终F1提升不能单独证明 hindsight 归因正确。
