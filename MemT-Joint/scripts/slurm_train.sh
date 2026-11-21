#!/bin/bash
#SBATCH --job-name=memt-joint
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/MemT-Joint
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-%j.err
set -euo pipefail
ROOT=/public/home/wwxia/yuang.li
[[ "$PWD" == "$ROOT/workspace/MemT-Joint" ]]
[[ "$(stat -c %u "$ROOT")" == "$UID" ]]
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
set +u
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/miniforge3/envs/memt-joint"
set -u
[[ "$CONDA_PREFIX" == "$ROOT/miniforge3/envs/memt-joint" ]]
export PYTHONPATH="$PWD:$PWD/third_party/Tree-GRPO"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="$PWD/artifacts/tmp-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
ulimit -c 0
nvidia-smi
if [[ "$#" -eq 0 ]]; then
    echo 'Pass CLI arguments, e.g. train --config configs/joint.json --data ... --valid-data ... --output ...'
    exit 2
fi
if [[ "$1" == sft ]]; then
    exec torchrun --standalone --nnodes=1 --nproc_per_node=4 cli.py "$@"
fi
exec python -u cli.py "$@"
