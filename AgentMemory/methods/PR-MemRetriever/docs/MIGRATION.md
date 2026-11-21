# 根目录整理与历史复核

本地：`D:\Desktop\工作文件\Mem-T`；远程：`/public/home/wwxia/yuang.li/workspace/Mem-T`。只整理这两个项目目录。

| 原位置 | 当前位置/处理 |
|---|---|
| `memt/{models,memory,methods,training,evaluation,experiments}` | 对应根目录包，实际文件迁移 |
| `memt/cli.py` | `cli.py`；入口`python cli.py`或`python -m cli` |
| `memt_training/` | 删除重复兼容别名，调用者改为根目录包 |
| 根目录`memory_formation.py`等 | 有效协议整理入`memory/` |
| `llm_judge.py`中的两份LoCoMo提示词 | 原样提取至`evaluation/prompts.py`，未改变判分语义 |
| `main.py`、旧builder/HTTP客户端/服务入口 | 无当前调用后删除；源码快照保留原始内容 |
| `data/ database/ logs/ traj/ models/README.md` | 删除未使用占位说明；真正数据和权重位置不变 |
| `assets/`、旧长程作业启动器、旧完整源码副本 | 从活动源码移除，按整理前快照归档 |
| `docs/validation/`与旧`docs/VALIDATION.md` | 迁入`artifacts/validation/`，原始记录不改写 |
| `requirements-training.txt` | 成为唯一主依赖清单`requirements.txt` |

整理前原始文件封存在`artifacts/code-snapshots/root-layout-before-20261102/source.tar.gz`，同目录记录文件SHA清单和删除/映射计划。远程原有`artifacts/`条目逐项核对大小和mtime不变；只新增本次快照、验证报告及布局审计文件。

历史代码中可能出现旧包名、旧根目录文件及旧报告相对链接，这是历史记录。复核旧模型时使用其对应源码快照；当前架构入口以根目录README为准，不将历史manifest改写成新布局。

新的构建短程配置输出到`artifacts/construction-smoke4-root-layout`。旧`artifacts/construction-smoke4-20261102`、检索200步及原始权重对照结果保持不变。本次只做代码迁移与CPU回归，不自动重训。
