# MemT-Joint

- 本项目实现已确认的四类记忆、B/U Global+Local、R固定库MoT共享联合RL。
- 用户原始Mem-T目录只作来源参考，不修改其文件、模型或历史结果。
- fixture只验收环境；不能将它的输出标成Qwen训练或真实数据集结果。
- 修改采样/分组/损失/恢复时运行对应CPU测试，保留真实模型尚未验证的边界。
- 远程仅操作`/public/home/wwxia/yuang.li`下自己的文件、环境和进程；不用sudo或系统Python。
- 新项目使用个人Miniforge的独立`memt-joint`环境；基础权重仍放个人`models/`。
- 运行产物使用新run目录；不覆盖旧结果，不复制样本凑数量，不把测试题用于训练/选模。
