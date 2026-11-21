# Mem-T v2 实现约定与未公开细节

本项目为研究重实现。上游 commit 见 `UPSTREAM.lock.json`，不声称恢复了作者未发布的训练源码。

| 项目 | 依据或约定 |
| --- | --- |
| LoCoMo 划分 | 上游归档中的 main.py：原始顺序第 1 个对话训练，第 2 个验证，其余测试；不随机划分 QA；排除 category=5 |
| 检索树 | 论文 §3.2：G=3、深度4、每轮3个 pivot；合成根不参与奖励和优势统计 |
| 扩展轮数 M | 论文未明确；配置不填写，启动命令必须显式指定；建议先用1并比较1/2 |
| 证据系数 alpha | 论文未明确；配置不填写，启动命令必须显式指定；建议比较0/0.5/1 |
| 标准差 | 对节点总体使用 population std（ddof=0），零方差优势为0；上游 torch.std 默认样本 std，不能静默混用 |
| 证据集合 | 使用当前节点路径累计实际返回记忆的 source_turn_ids；不把未检索的 store 内容算作命中 |
| 未完成轨迹 | 深度耗尽时答案表现为0，不额外插入未采样的 finish；语法有效的检索节点仍可有证据奖励 |
| 无效调用 | 无效检索调用终止分支且 reward=0；数据库或资源错误直接失败，不伪装成策略格式错误 |
| pivot 采样 | 复用上游 random.choices 的有放回语义；从 pivot 执行后的状态继续生成 |
| 节点损失 | 每个唯一采样节点出现一次；节点内 token 均值、树内节点均值、跨树均值、跨问题均值 |
| GRPO 对照 | 独立链数量为 G*(1+M*Nv)，对应树采样的完整 rollout 次数上界；实际 token 成本另行记录，不能声称严格 token 等预算 |
| 跨问题归因 | 每个问题先对全部叶子取平均，再对该 episode 的问题取平均；不混用训练中不同 checkpoint 的森林 |
| 记忆版本 | UPDATE 生成新版本，继承来源与生产操作；检索新版本时给其保留的生产链 trace credit，这是 lineage 约定，不是因果证明 |
| DELETE/IGNORE | 保留操作与来源；没有输出记忆的操作只能通过 evidence gate 获取分数，不伪造其输出记忆 |
| 原子动作 | 一次 completion 可包含多个构建工具调用；按调用分类、排名；每个入选调用只监督自己的 token span，首个调用包括其前置 reasoning |
| top50% | 每个 phase:tool 类别按分数排序，ceil(N*0.5)；并列用 op_id 稳定打破，记录全同分现象；负分也按原文排名选取，不另增正分门槛 |
| 10k 构建操作 | 作者未公开生成流水线；不复制样本凑10k。rollouts 参数可采集独立构建轨迹，实际数量和类别分布留痕；超过显式 cap 报错 |
| 两阶段权重 | 默认构建和检索各从相同 base 初始化；先训练检索并固定该策略，再训练构建；评测两种角色使用各自 checkpoint。作者最终权重共享/合并流程不明 |
| 训练后记忆库 | 必须使用训练后的构建策略重新生成测试库；不能继续使用训练前缓存 |
| Raw archival | legacy-v1按公开协议自动归档；paper-v2增加可学习create_raw(turn_ids)，校验当前块真实ID并复制原文，两种协议不混合 |
| 构建训练后端 | 原子 mask 使用 Transformers Trainer + DeepSpeed ZeRO-3；另提供 LLaMA-Factory 导出。后者只保留全部调用均入选的 completion，导出审计说明覆盖差异 |
| 上下文预算 | 检索超预算明确失败，不截断后仍把未显示证据算作命中；SFT 按6144保留动作和最近上下文 |
| 生成分布 | RL 使用 temperature，禁用 top-p/top-k/min-p 截断；旧策略和参考策略按相同 temperature 计算 logprob |
| 推理设置 | greedy，最多6个动作，top5；属于显式选择，作者代码中 temperature 有不一致 |
| 推理 fallback | 沿用公开实现，未 finish 时额外调用一次 AnswerWithMemoriesTool；单独记录 fallback 与 token，不把它算作训练树中的额外叶子 |
| Qwen3上下文 | 基础模型 config 原生总窗40960；1024生成预算时实际prompt上限39936。记录effective_prompt_tokens，不隐式开启超窗/RoPE扩展 |
| NarrativeQA | 精确10篇文档清单未公开；本适配采用seed42选10篇并留清单、512词分块，不宣称精确重建298题 |
| LongMemEval | 本地提供逐题 F1 与轨迹；论文 Accuracy 应通过官方 LLM judge 另算，不能把 F1 改名为 Accuracy |

后续研究应分别报告“论文明确配置”“上述实现假设”“实际运行配置”，以及每项消融的实际生成 token 和 wall time。
