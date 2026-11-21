# 实现与原始复现的对应关系

来源为用户本地`D:\Desktop\工作文件\Mem-T`的活动源码，原目录未修改。`SOURCE_ORIGIN.json`记录逐文件来源SHA；`UPSTREAM.lock.json`保留上游声明。新代码和修改文件在运行manifest中重新计算SHA，不能仅凭上游commit声称代码相同。

## 实际实现

| 功能 | 位置 |
|---|---|
| 四类存储、独立快照与B/U事务 | memory/branch.py，复用formation/update/Journal工具 |
| 真实显示来源、提交证据、只读检索 | methods/joint/retrieval.py |
| Global、B/U Local、greedy后续续跑 | methods/joint/rollout.py |
| reward-to-go、Local增量、MoT优势、角色权重 | methods/joint/credit.py |
| step-ratio与token裁剪 | methods/joint/loss.py |
| 当前 policy 直接 finish(answer) | models/joint.py、methods/joint/retrieval.py |
| 共享HF更新、完整CPU状态恢复 | training/native.py |
| 共享四卡后端与actor | training/verl_backend.py、verl_actor.py、third_party/Tree-GRPO |
| 独立参数对照 | training/separate.py |
| 两阶段真实采集、回溯与原子SFT | experiments/two_stage.py、training/sft.py |
| 训练、选模、恢复、身份、成本 | experiments/joint.py |
| 离线judge与覆盖检查比较 | evaluation/joint_judge.py、comparison.py |

## 必须公开的适配

- 持久状态使用可深克隆的collection与JSON快照，复用Mem-T的Journal/工具；真实配置从空状态开始即使用Chroma+BGE-M3，fixture才使用词面后端。没有把测试用词面排序冒充BGE，也没有声称与Chroma的ANN索引逐位等价。所有共同协议主对照使用同一后端，原协议结果单列。
- B的一次completion可含多操作；整次调用只进入损失一次。格式/执行失败的completion不提交其部分副作用。U只允许修改已展示候选ID。
- 标准LoCoMo终态QA不提前变成在线题。早期Local可顺序续跑greedy后续角色到终态，只给候选自身梯度。可用`current_only`检查无续跑版本；原生终态数据会跳过早期anchor。
- R的finish直接提交当前 policy 生成的答案；fallback 仍只使用该路径已显示的证据，不调用冻结 Answer。
- R森林按问题/树的活动节点批量生成，唯一节点出现一次；不同库、不同题不混优势组。Global R不进入主方法R损失。
- Working来源、事实文档来源、metadata复制都进入可检索存储成本；版本审计与完整源对话另行记录。保留Raw少不等于物理总存储必然下降。
- 40K协议下配置 Prompt 上限39936、生成1024。R在下次生成前检查完整带角色Prompt，超长路径记为prompt_budget失败；超预算不截断，也不把不可见来源算命中。
- 训练checkpoint保存完整模型/优化器/调度器与RNG；Native另保存冻结参考。分布式状态依赖固定基础参考身份和上游FSDP保存接口。恢复先核验receipt哈希，source/model/data/config漂移拒绝续跑。
- 原文关闭诊断重新运行检索并禁用Raw工具；固定原轨迹只移除Answer原文不是完整Memory-only。
- M5与共享HF控制组比较参数共享；不将后端差异隐藏为共享收益。M6使用相同新Answer/证据协议，故不是原论文指标的无修改复现。
- 主线默认200共享更新、R目标32组；它不等同于原方法200 RL加200 SFT的计算量。实际调用、token、含保存验证的wall time需同时报告。

## 许可与外部评测

Mem-T与Tree-GRPO的许可证保留在根目录和`third_party/Tree-GRPO/LICENSE`。LongMemEval题型prompt源于其[官方评测代码](https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)，MIT通知在`third_party/LongMemEval.LICENSE`。本实现采用严格完整yes/no解析并报告judge模型，不将任何judge分数冒称其他模型结果。

## 验证记录

`python -m scripts.check`写入`artifacts/validation/LOCAL_CHECKS.json`。其中fixture只验收控制流；微型Torch策略确实采样、反向、更新、停止、恢复并验证选模，词表为合成工具token。该证据覆盖CPU机制和状态恢复，不覆盖Qwen Tokenizer、真实4B显存、Ray/vLLM四卡、真实SFT或benchmark效果。
