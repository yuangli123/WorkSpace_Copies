# 评测线程资源修复（2026-11-10）

作业 251794 已按用户要求取消；保留原 run、step50 导出权重和 Ray 日志。
本次不提交新的完整评测。

## 证据与原因边界

- PROGRESS 停在 conv-30 的 6/99 块，尚无验证分数。
- 两张 GPU 利用率为 0%，主进程等待 futex，Ray worker 等待 epoll。
- 多个 Ray worker 在 16:41:34 报 `thread: Resource temporarily unavailable [system:11]` 并 SIGABRT。
- 原 ChromaBranchDB 每次克隆都创建新临时路径和 PersistentClient。实际环境 Chroma 1.0.9 的 SharedSystemClient 按路径全局缓存 System，Python 对象回收不释放这些运行时。
- Ray 原先检测到节点 48 CPU 并预启动 48 worker，而 Slurm 只分配 12 CPU。
- 上述证明线程资源失败和客户端累积缺陷；没有取得原主线程 Python 调用栈，不能把具体阻塞位置断言为某次 ray.get 或 Chroma 调用。

## 最小修复

- 每个 runtime 的 ranker 共享一个 Chroma 客户端；分支使用独立物理集合，snapshot 仍使用原逻辑集合名。
- 分支回收时清理对应集合；HNSW num_threads=1。
- 关闭 Chroma 匿名遥测，避免离线节点反复请求 PostHog。
- 本地 Ray 实例按 SLURM_CPUS_PER_TASK 限制 CPU；连接外部 Ray 时不覆盖集群资源。
- 评测启动脚本关闭 worker 预启动，并限制 BLAS/MKL/Rayon 线程数。

## 验证

- 私有 memt-joint 环境：2 项生命周期测试通过。
- 模拟库连续 100 次克隆：客户端始终 1，分支互不污染，清理后无残留集合。
- 真实 Chroma 连续 40 次克隆、写入、搜索：客户端 1，存活集合回到 5，线程 54 → 55；metadata source_turn_ids 正常回读。
- 真实 Chroma 测试使用确定性小向量函数隔离资源问题；未加载 BGE-M3/Qwen 或重跑 81 题。
- 远程 Python 编译和 bash -n 通过；同步文件哈希与本地一致。
- 扩展回归最终 31 项通过（37.64 秒），覆盖共享记忆、统一协议、信用分配及 Chroma 生命周期；日志 `regression-final.log`。补齐测试副本依赖，并修正原直接答案测试遗漏的 state 初始化。

远程补丁位置：`/public/home/wwxia/yuang.li/workspace/MemT-Joint-eval-nonfrozen-20261110`。
改前文件及回归日志：该副本的 `artifacts/thread-fix-20261110/`。
原 Mem-T 和历史训练 run 未修改。
