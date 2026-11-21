#!/bin/bash
#SBATCH --job-name=memtj-cap40k
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/MemT-Joint
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-%j.err
set -euo pipefail
ROOT=/public/home/wwxia/yuang.li
[[ "$PWD" == "$ROOT/workspace/MemT-Joint" && "$(stat -c %u "$ROOT")" == "$UID" ]]
set +u
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/miniforge3/envs/memt-joint"
set -u
[[ "$CONDA_PREFIX" == "$ROOT/miniforge3/envs/memt-joint" ]]
export PYTHONPATH="$PWD:$PWD/third_party/Tree-GRPO"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0
export TMPDIR="$PWD/artifacts/tmp-$SLURM_JOB_ID" RAY_TMPDIR="$ROOT/r$SLURM_JOB_ID"
RUN="$ROOT/workspace/AgentMemory/artifacts/memt-joint/common40k-capacity-job$SLURM_JOB_ID"
[[ ! -e "$RUN" ]]
mkdir -p "$RUN" "$TMPDIR" "$RAY_TMPDIR"
export XDG_CACHE_HOME="$RUN/cache" TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions" HF_HOME="$ROOT/models/huggingface"
ulimit -c 0
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python -m scripts.check
exec python -u -m scripts.profile_common --config configs/common-a100x2-50.json --output "$RUN"
