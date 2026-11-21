# 正式构建SFT与两组完整测试

入口：`python -m experiments.study_dispatch --config configs/experiments/construction-study-200.yaml`。
只在远程个人项目根目录、自己的Conda环境执行。模型与结果不迁移、不覆盖。

## 训练与对照

从原始Qwen3-4B采样完整训练对话；每个rollout使用不同sample ID和种子。
累计至少24000条有效原子动作后，在完整rollout边界停止，最多80个rollout。
24000是筛选前的采样停止阈值，为类别内前50%筛选、超长动作剔除和验证划分留余量；
最终实际采样、入选、训练和验证数量分别报告，不复制旧动作补数量。
全部训练QA（152题/rollout）使用同一冻结R200权重采集森林。
沿用三树、深度4、三个pivot、一次扩展、证据系数1、trace权重0.1；
先对每题叶子平均，再对每个episode的问题平均，统一按动作类别保留前50%。
如果不足10000条入选动作或归因退化，停止诊断，不开始SFT。

SFT：原始Qwen3-4B初始化，200步，4卡×每卡batch2×梯度累积4=全局32，
BF16、ZeRO-3、FA2、max_length6144、lr5e-6、cosine、warmup0.1、seed42。
只监督入选动作，原始completion分组划分90/10；只导出最终HF权重。
每步loss/梯度/实际学习率写入construction/metrics；完整权重重载后重新处理全部测试对话。

依次评测1307题：

1. construction_only：C200构建记忆，原始Qwen3-4B检索。
2. construction_retrieval：同一份C200记忆，已有200步检索权重检索。

两组记忆快照、QA顺序、解码参数和评审模型相同，检索器不再训练。
这组对照可衡量C200固定时检索训练的增益，不能单独证明构建训练相对C0的增益。
输出F1、BLEU-1、judge、完成率、逐题配对记录及模型/源码/数据身份。

## Judge口径

用户最终选择Memory-R2宽松原文，版本memory-r2-generous-v1。
原文位于evaluation/prompts.py，不混用之前讨论的严格规则。
按原文允许同主题/日期即CORRECT；因此不能解释为严格事实准确率。
JSON必须含label=CORRECT或WRONG，reasoning可选，以兼容原文最后的label输出要求。
先运行三个明确控制题及一个时长不一致诊断题；诊断题不参与1307题分母。
历史legacy-topic-v1提示词及历史结果保持不变；两个宽松版本也需要分别标记。

## 作业、恢复与预算

运行目录artifacts/construction-study200-20261102；CURRENT.json、SUBMITTED.json、stages/*.json及实际Slurm状态共同核对。
采样保存每个完整输入块及episode校验；完整episode恢复不重复嵌入。
森林、评分和动作数据gzip压缩、逐题恢复；评测保存逐题seal并检查输入/模型/源码身份。
只有接近6小时Slurm时限的明确deferred状态会提交afterok后续作业；
运行错误、预算错误和观察超时不会触发自动重试。最多32个分段作业。
启动SFT前至少预留4小时及10分钟余量；若SFT被时限中断，因无优化器检查点需从原始权重重训，最多两次尝试。
不宣称真实4B优化器续训已验证；微型模型完整恢复的历史证据单独保留。

100GB按十进制、inode去重、max(apparent,allocated)口径；90GB预警，既有Conda环境单列排除。
首次准入预留26GB新增数据/权重/缓存空间，后续按实际台账逐阶段和写入前检查。
共享缓存只读审计；不会自动删除旧实验、旧权重或修改其他作业。
新实验的Chroma集合显式使用HNSW num_threads=1，避免大量独立rollout的集合线程叠加；旧数据库不改写配置。

验收以COMPLETE.json、comparison.json、REPORT.md为准，提交作业不代表已完成训练或正式测试。
