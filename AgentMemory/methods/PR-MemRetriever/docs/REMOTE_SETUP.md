# 个人环境与存储记录

使用现有`/public/home/wwxia/yuang.li/miniforge3/envs/memt-train`，禁止系统Python、sudo或其他用户环境。

```bash
source /public/home/wwxia/yuang.li/miniforge3/etc/profile.d/conda.sh
conda activate memt-train
cd /public/home/wwxia/yuang.li/workspace/Mem-T
python cli.py --help
```

当前训练使用Python3.11、PyTorch2.6.0、Transformers4.51.3、vLLM0.8.5.post1、DeepSpeed0.16.7、FlashAttention2.7.4系列；环境锁记录保留在原`artifacts/setup/`。主依赖清单统一为`requirements.txt`。本轮目录整理不执行安装或升级。

未来重建环境时，先按当前GPU/CUDA环境安装匹配PyTorch，再安装主依赖与匹配的FlashAttention wheel；CentOS7/glibc2.17的历史构建说明可在整理前源码快照内查看，不直接套用其他CUDA版本的wheel。

`scripts/slurm_train.sh`申请本人的4卡资源，激活已有环境；缓存和日志在`artifacts/`，短TMPDIR在项目根目录下，避免vLLM Unix socket路径超长。Conda激活阶段暂时关闭Bash nounset。NPROC只设置新作业内自身的软限制，不改其他进程或系统配置。

模型：个人`/public/home/wwxia/yuang.li/models/{Qwen3-4B,bge-m3}`；实验输入、记忆、训练权重及结果：项目`artifacts/`。根目录`models/`是Python包。预算沿用十进制100GB口径，既有Conda环境单列排除。
