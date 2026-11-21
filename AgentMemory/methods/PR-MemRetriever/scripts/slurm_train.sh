#!/bin/bash
#SBATCH --job-name=memt-train
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/Mem-T
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/Mem-T/artifacts/slurm-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/Mem-T/artifacts/slurm-%j.err
set -euo pipefail
[[ "$UID" == 2086 && "$PWD" == /public/home/wwxia/yuang.li/workspace/Mem-T ]]
ulimit -S -u 16384
ulimit -c 0
set +u
source /public/home/wwxia/yuang.li/miniforge3/etc/profile.d/conda.sh
conda activate memt-train
set -u
RUN="$PWD/artifacts/runtime/slurm-$SLURM_JOB_ID"
if [[ "${1:-}" == run-construction-study ]]; then
    RUN="$PWD/artifacts/runtime/construction-study"
fi
export TMPDIR="$PWD/.cs-$SLURM_JOB_ID" XDG_CACHE_HOME="$RUN/runtime/cache"
export TORCH_EXTENSIONS_DIR="$RUN/runtime/torch_extensions" TRITON_CACHE_DIR="$RUN/runtime/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN/runtime/inductor" VLLM_CACHE_ROOT="$RUN/runtime/vllm"
export HF_HOME="$RUN/runtime/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=2
export ANONYMIZED_TELEMETRY=False
export CUDA_CACHE_PATH="$RUN/runtime/cuda"
if [[ -d /usr/local/cuda-12.2 ]]; then
    export CUDA_HOME=/usr/local/cuda-12.2
    export PATH="$CUDA_HOME/bin:$PATH"
fi
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$HF_HOME"
[[ "$CONDA_PREFIX" == /public/home/wwxia/yuang.li/miniforge3/envs/memt-train ]]
[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]
RECORD="$RUN/jobs/$SLURM_JOB_ID"
mkdir -p "$RECORD"
scontrol show job "$SLURM_JOB_ID" -o > "$RECORD/allocation.txt"
nvidia-smi > "$RECORD/gpu-at-start.txt"
if [[ "$#" -eq 0 ]]; then
    set -- run-construction --config configs/experiments/construction-smoke4.yaml --resume
fi
if [[ "${1:-}" == run-construction-study ]]; then
    set +e
    python -u -m cli "$@"
    code=$?
    set -e
    if [[ "$code" == 75 ]]; then
        [[ "${2:-}" == --config && -n "${3:-}" ]]
        python -u -m experiments.study_dispatch --config "$3" --after-job "$SLURM_JOB_ID"
        exit 0
    fi
    exit "$code"
fi
exec python -u -m cli "$@"
