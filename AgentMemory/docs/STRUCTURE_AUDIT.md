# 本地目录整理审计（2026-10-02）

操作范围仅 `D:\Desktop\工作文件\AgentMemory`，未连接或修改远程服务器。移动前确认 `RecMem`、`MemoryOS`、`mem0`、`A-mem` 在工作区根目录；移动后四个目录完整位于 `methods/`，三个 `.git` 保留。没有新建顶层 Git 仓库；`mem0/evaluation` 子模块仍在，提交 `4b61c5d31b9c`。

| 仓库 | 移动前 HEAD / 工作区状态 | 移动后核对 |
| --- | --- | --- |
| RecMem | `a84252f6e558`；原有 README、评分工具、运行入口、embedding 和 RecMem 源码未提交修改，另有外部基线与测试文件 | 同一 HEAD；这些源码改动仍在；`.env.example` 从仓库移到工作区根目录，因此仓库状态显示 `D .env.example` |
| MemoryOS | `587ed7755c7a`；`eval/main_loco_parse.py`、`mid_term_memory.py`、`utils.py` 未提交修改 | 同一 HEAD、同三处未提交修改 |
| mem0 | `989c7da0fc8e`；干净 | 同一 HEAD、干净；`evaluation` 子模块内容保留 |
| A-mem | 现有下载源码，无 `.git` | 整体保留在 `methods/A-mem`；未裁剪内部文件 |

移动后的文件清单摘要（含 `.git`、测试和可清理 Python 缓存；具体文件可在各目录按需列出）：`methods/RecMem` 101、`MemoryOS` 99、`mem0` 1956、`A-mem` 19；`docs/references` 3；本地 `artifacts/{data,results,logs,stores,cache}` 在整理时均为空。本地 `.env` 已存在于工作区根目录，本轮内容原样保留，未打印或提交。

关键文件 SHA-256：

```text
recmem.txt                         9E2821314EF2BA88C54200377DDB5A94610DFF53512EC696508155B24B82B658
RecMem_ACL2026.pdf                 0E9CE858E3812378F18299899DA53DF2B16D18674E16862D7D1F7528CFD3339B
RecMem/recmem/rec_mem.py           9D86A59C17C67359E089120AB2F38FA273EEA12149B79C93ABCB71231E05076B
MemoryOS/eval/main_loco_parse.py   161CCB2387F83E2C8BC04D01FADFCAB5B9A508B10A94AFC0CC1C86B22FF075E9
mem0/mem0/memory/main.py           F63B856FF86B2B9DA3730548116BFBC023560FE941DBFBD34075519B855C46D3
A-mem/agentic_memory/memory_system.py 4B5B3BA8FF167242121C4C00F6B463D7582A71D8A137E877816AB7CB7C83EC17
```

整理前 `paper.txt` 与保留的 `recmem.txt` SHA-256 同为 `9E2821...82B658`，故仅删重复的 `paper.txt`；原 PDF 和另外两个参考文本都保留在 `docs/references/`。另删除了 `tmp/pdfs/recmem-page-01.png` 至 `05.png` 五张可由 PDF 再生的页面渲染图及空 `tmp` 目录。Python `__pycache__`/`.pyc` 是可再生缓存。未删除数据集、模型、上游文件或实验结果。

验收：整理前离线测试 20 项通过；整理后增加迁移测试，22 项通过。`external_runner`、`compare_results`、`migrate_layout` 的 `--help` 可在本地直接运行；`run_experiments` 的参数帮助通过离线依赖替身验证，因本地当前 Python 缺少 `python-dotenv`，直接 `python -m` 的帮助命令尚待配齐环境后复查。远程同步、迁移和实际模型联调均未在本轮执行。
