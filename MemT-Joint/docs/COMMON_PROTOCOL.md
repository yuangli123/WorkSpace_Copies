# 统一比较协议

当前执行协议为 **memt-execution-v2**，以 [MEMT_EXECUTION_ALIGNMENT.md](MEMT_EXECUTION_ALIGNMENT.md) 为准。
下文保留历史 memt-common-v1 说明；其中自定义提示词、JSON cards、累计路径硬限制和证据装包 fallback 已由 v2 替换，不能用于描述当前代码。

## 历史协议 memt-common-v1

本协议用于比较训练策略。原 Mem-T 源码不改动，旧实验结果单列。MemT-Joint 的端到端评测从空状态构建记忆，并在检索阶段使用与 Mem-T 一致的 Chroma 语义。

| 条件 | 全部策略共同设置 |
|---|---|
| 策略模型 | Qwen3-4B；B/U/R 均由 vLLM 0.8.5.post1 V0、TP=2 生成，禁用前缀缓存、CUDA graph、自定义 all-reduce |
| Answer | 不再有独立 Answer；检索 policy 使用当前共享权重直接生成 `finish(answer)` |
| 上下文 | 输入 39,936 + 输出最多 1,024 = 总计 40,960 tokens；观察最多 20,480；检索路径成本预算 40,960 |
| 评测采样 | greedy，temperature=0，top_p=1，top_k=-1，min_p=0；检索最多 6 步 |
| 存储 | paper-v2：Builder 选择 Raw；Working/Factual/Experiential/Raw；相同 BGE-M3 排序器与 top-k |
| 工具 | 相同的完整参数 schema、选择说明、闭合标签示例；finish 接受 `answer` 并直接结束检索 |
| 答案 | 按 Mem-T 的 benchmark/category 规则输出短语；优先原文措辞；日期如 `15 July, 2023`；相对时间依据原文 turn_time；LoCoMo 不输出 Unknown |
| 失败 | 严格校验动作；未正常 finish 时，同一当前 policy 仅使用该路径已展示且能放入预算的证据 fallback |
| 指标 | 主 F1/BLEU-1、strict F1/BLEU-1、finish/fallback 比例、实际证据覆盖；非法 R 动作仍获 invalid 惩罚 |

“共同后端”是所有角色共享当前 policy 的 vLLM 推理；没有冻结 Answer 权重。M5 的梯度更新仍为 HF 独立参数；生成共用一个 TP 引擎，更新后刷新，不合并角色参数。使用的始终是本作业获分配的两卡，训练与生成按阶段串行。

共享的提示词位于 `methods/joint/prompts.py`。每个结果保存 protocol_id，包含提示词、预算、后端版本、当前 Answer 权重身份等；compare 拒绝混合协议。训练角色、奖励分支权重不进入推理协议指纹。历史 55.45% 与旧版 5.45% 不能当作本协议下的训练增益。

## 执行顺序

1. `python -m scripts.check`：CPU/fixture 检查，不证明显存可行。
2. 两张 A100 40GB 上执行 `python -m scripts.profile_common --config configs/common-a100x2-50.json --output <独立目录>`：接近 39,936-token 输入的当前 policy 生成，以及固定 1,024-token 响应的两次连续 old/ref 概率计算、反向传播和 Adam 更新，第二次包含已有优化器状态。另执行 `python -m scripts.profile_bridge --output <另一个独立目录>`，检查真实权重刷新和退出清理；要求 Slurm 正常退出，不能只看成功 JSON。合成更新用后丢弃，不能计入真实 RL step。
3. 容量检查通过后，新进程从基础权重执行 `python cli.py train --config configs/common-a100x2-50.json --variant M4 --data <train> --valid-data <valid> --output <新目录>/train`。先用同一运行时计算完整 81 题 step-0 基线，再做 50 次真实共享更新，每 10 步保存及验证。不会复用旧协议基线。
4. 其他对照沿用 `configs/common-memt-v1.json` 与 `--variant M0..M4`；独立参数 M5 用 `configs/common-m5.json`。M6 用 `configs/common-m6.json`，保留既有两阶段流程。训练后评测须提供对应导出权重。M5/M6 的真实长上下文训练另需硬件验收，M4 容量通过不代替它们的验收。

`train/BASELINE.json` 指向新协议基线；`train/validation/step-*.json` 指向各轮完整评测；`PROGRESS.json` 显示逐块建库/逐题检索进度。显存峰值为资源监控的采样值，可能低于瞬时峰值。容量检查的生成可能提前 EOS，报告实际输出长度；反向更新固定覆盖 1,024 个响应 token。

不降低预算、不删除角色或 Local 分支来绕过显存失败。若 40K 失败，保留失败阶段与资源日志，先定位实际内存分配，再决定最小修复。

40K 显存修复保留完整注意力上下文，只对响应位置做词表投影。已有 Adam 状态在反向传播结束后才从 CPU 载入 GPU，避免与激活峰值叠加。FSDP 返回的 loss/KL 显式跨 rank 汇总，避免将 rank 0 的局部统计当作全局损失。
