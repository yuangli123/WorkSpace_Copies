# Mem-T 执行对齐：memt-execution-v2

本版本保留 paper-v2 原文选择和空状态按时间构建；共享 B/U/R、Global+Local、R MoT 及损失公式保持不变。

## 执行契约

- 检索直接使用 Mem-T 的 MemoryRetriever、工具 schema、category-specific prompt、观察格式和 seen_turns 去重。
- policy 直接输出 finish(answer)，fallback 复用 AnswerWithMemoriesTool 和本路径 observation（包括 Mem-T 的非法动作反馈），使用当前 policy。动作错误文本不计作原文证据。
- B/U/R 的角色仅留在调度与训练日志，不写入模型提示；persona context、候选 turn_time、原文分组与 Mem-T 对齐。
- 构建逐操作执行：保留合法操作，记录非法操作；completion 仍可因包含非法操作获得既有 invalid 惩罚。
- 输入 39936、输出 1024、单次 observation 20480；不静默删减原文，不以累计路径成本作为额外停止条件。
- 路径成本继续用于现有奖励。数据库、后端和预算异常向上传播，不计作模型非法动作。
- Chroma 增量同步、嵌入缓存、历史状态隔离保留；memory_only 诊断通过只读视图隐藏原文，不改变主路径。

Mem-T 参照文件哈希在 MEMT_EXECUTION_REFERENCE.json；工具实现仅适配项目 logger，并让 latest-summary 的数据库异常上抛，避免伪装为空结果。正常路径的类和函数 AST 由差分测试核验。
与历史协议不兼容：必须新建 run，不能恢复旧 manifest 或复用 baseline。

## 验收

1. `python -m scripts.check`：同输入对照、工具边界、部分非法操作、证据展示、时序及训练回归。
2. `python -m scripts.verify_memt_backend --model <本地BGE-M3> --output <独立目录>`：真实 BGE/Chroma 的 top-k、顺序、metadata 与历史分支切换。
3. `scripts/slurm_memt_execution_step0.sh`：两张 A100 40GB、基础 Qwen3-4B、固定81题 valid、从空状态建库，无训练更新。

GPU 输出包括完整 manifest、summary、逐题 answers、检索 trajectories 和 step0 完成回执。F1、BLEU-1、strict、finish/fallback 及分类指标均保存。
临时文件实际保存在个人 workspace/MemT-Joint/artifacts；Ray 的短路径链接位于个人 workspace，作业退出时仅移除本作业链接。

## M4 50步训练

使用 `configs/memt-execution-m4-50.json` 和 `scripts/slurm_memt_execution_50.sh`，先等待新版独立 step0 验收成功；训练自身仍从基础权重和空记忆重新执行 step0，不复用独立评测快照。
每10步保存并完整验证81题；保留最新可恢复检查点、验证集最佳模型和全部逐步指标/验证记录。新检查点完整写入后才轮换旧权重；最佳模型通过硬链接独立保留，历史元数据及轮换审计不删除。
单个 Slurm 分配为两张 A100、最长3天。距离时限18小时发出停止请求，当前步完成后保存；仅在正常完成时限检查点且未达到50步时继续同一run。不因OOM或代码异常自动循环重试，不响应用户手动STOP进行续交。
