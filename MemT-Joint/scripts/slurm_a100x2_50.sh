#!/bin/bash
#SBATCH --job-name=memtj-50x2
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/MemT-Joint
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-%j.err
set -euo pipefail
ROOT=/public/home/wwxia/yuang.li
[[ "$PWD" == "$ROOT/workspace/MemT-Joint" && "$(stat -c %u "$ROOT")" == "$UID" ]]
[[ -n "${SLURM_JOB_ID:-}" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]
set +u
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/miniforge3/envs/memt-joint"
set -u
[[ "$CONDA_PREFIX" == "$ROOT/miniforge3/envs/memt-joint" ]]
export PYTHONPATH="$PWD:$PWD/third_party/Tree-GRPO"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="$PWD/artifacts/tmp-$SLURM_JOB_ID"
export RAY_TMPDIR="$ROOT/r$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
ulimit -c 0
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full"
RUN="$ROOT/workspace/AgentMemory/artifacts/memt-joint/a100x2-50-20261106-job$SLURM_JOB_ID"
[[ ! -e "$RUN" ]]
mkdir -p "$RUN"
python -c 'import torch; assert torch.cuda.device_count()==2; print("ALLOCATED_GPU_COUNT", torch.cuda.device_count(), flush=True)'
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python -m pip check
python -m scripts.check
python cli.py preflight --config configs/a100x2-50.json --data "$DATA/train" > "$RUN/preflight.json"
python -u cli.py evaluate --config configs/a100x2-50-eval.json --variant M0 \
  --data "$DATA/valid" --output "$RUN/step000-eval"
exec python -u cli.py train --config configs/a100x2-50.json --variant M4 \
  --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUN/train"
