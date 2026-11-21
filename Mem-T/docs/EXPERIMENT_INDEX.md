# 模型与实验索引

| 实验 | 结果/配置位置 | 状态 |
|---|---|---|
| 检索200步 | `artifacts/long200-20261027/retrieval` | legacy-v1；完成，权重保留 |
| 原始Qwen3-4B检索器对照 | 同run的`baseline-qwen3-4b-20261102` | legacy-v1；1307题完成 |
| 构建4步原实验 | `artifacts/construction-smoke4-20261102` | paper-v2；四卡batch32闭环完成，8题验证 |
| 新布局构建短程配置 | `configs/experiments/construction-smoke4.yaml` | 新输出`artifacts/construction-smoke4-root-layout`，本次不启动 |
| 构建正式200步 | `configs/experiments/construction-200.yaml` | 未执行，需峰值预算准入 |
| 四组件对照 | `configs/experiments/components.json` | 配置交付，未全部执行 |

旧协议1307题：基础检索器F1=38.4357%、BLEU-1=33.8509%、宽松judge=70.3137%；200步检索器分别为49.1992%、43.3853%、82.9380%。

paper-v2构建4步固定8题：F1 29.17%→38.45%，BLEU-1 29.17%→34.68%，宽松judge两组均87.5%。原子操作有效率72.53%→99.11%，completion有效率52.12%→98.39%。这些是原布局下已经完成的结果，本次重组没有重新训练或改写它们。

原始逐题结果、权重、`COMPLETE.json`与独立审计留在各run中；本地此前同步的小型报告位于`artifacts/validation/construction-v2/`。宽松judge有明确误判，且部分F1增量来自词面重叠，不能用8题短程结果声称事实能力提高。旧协议结果不能直接作为新增CrtRaw后的构建基线。

四组件分别固定base/trained的construction和retrieval角色。两种构建模型必须分别处理相同完整验证对话；检索器、QA顺序、提示词与judge协议逐项一致。训练后必须重新建库。
