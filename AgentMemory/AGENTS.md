# AGENTS.md

本文件规范 RecMem 论文复现工作的本地同步与远程执行方式。

## 工作边界

- 本地工作区：`D:\Desktop\工作文件\AgentMemory`；各方法源码位于 `methods/`，统一复现步骤见根目录 `REPRODUCTION.md`。
- 远程实验根目录：`/public/home/wwxia/yuang.li`
- 远程源码位于 `workspace/AgentMemory/methods/`；数据、日志、存储、缓存和结果位于 `workspace/AgentMemory/artifacts/`。
- 远程下载的模型权重和 Tokenizer：只能放在远程 `models/` 下。
- 不得修改、删除或覆盖其他用户的文件、环境、进程或配置。
- 不得使用 `sudo`、系统级安装或系统 Python。

## 本地与远程同步

- 本地仓库用于同步源码、配置和文档；不要把模型权重、数据集或大文件提交到 Git。
- 远程实验产生的日志、检查点和结果保留在远程 `workspace/`，需要同步时只复制明确需要的文件到本地。
- 同步前确认目标是自己的 `workspace/AgentMemory`，禁止使用未检查的通配符或越界路径。
- 不确定文件归属、同步方向或覆盖风险时，先停止并确认。

## Python 环境

- 远程 Python 环境统一由 `/public/home/wwxia/yuang.li/miniforge3` 管理。
- 使用前加载：

  ```bash
  source /public/home/wwxia/yuang.li/miniforge3/etc/profile.d/conda.sh
  ```

- 每个项目使用独立的 Conda 环境；禁止使用系统 Python、系统 Conda、`sudo pip` 或 `pip install --user`。
- 修改依赖后更新项目环境记录，例如 `environment.yml` 或 `requirements.txt`。

## 远程节点安全

- 执行命令前确认当前路径位于 `/public/home/wwxia/yuang.li`。
- 实验前检查 `nvidia-smi`，不得终止或干扰不属于自己的进程。
- 禁止执行未经确认的 `rm -rf`、批量覆盖、权限修改或清理共享目录操作。
- 不在源码、日志或提交中保存密码、Token、SSH 私钥等凭据。

## 复现记录

每次实验记录代码版本、Conda 环境、数据集位置、模型路径、关键参数、完整命令和结果路径。

优先保持改动小而可验证：先复现当前问题，再做最小修改，最后运行对应检查并记录结果。
