#!/bin/bash
#SBATCH --job-name=memtj-short8-prof
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/MemT-Joint-short8-r3-20261113
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/short8-r3-profile-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/short8-r3-profile-%j.err
set -euo pipefail
ROOT=/public/home/wwxia/yuang.li
PROJECT="$ROOT/workspace/MemT-Joint-short8-r3-20261113"
[[ "$PWD" == "$PROJECT" && "$(stat -c %u "$ROOT")" == "$UID" ]]
set +u
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/miniforge3/envs/memt-joint"
set -u
[[ "$CONDA_PREFIX" == "$ROOT/miniforge3/envs/memt-joint" ]]
export PYTHONPATH="$PROJECT:$PROJECT/third_party/Tree-GRPO"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0
export RAY_enable_worker_prestart=0 RAY_prestart_worker_first_driver=0
RUN="$ROOT/workspace/MemT-Joint/artifacts/short8-r3-profile-job$SLURM_JOB_ID"
[[ "$RUN" =~ ^/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/short8-r3-profile-job[0-9]+$ ]]
if [[ -z "${TRAIN_RUN:-}" ]]; then [[ ! -e "$RUN" ]]; fi
mkdir -p "$RUN/runtime/job$SLURM_JOB_ID/ray" "$RUN/cache" "$RUN/continuations"
ALIAS="$ROOT/workspace/.q"
[[ ! -e "$ALIAS" && ! -L "$ALIAS" ]]
ln -s "$RUN/runtime/job$SLURM_JOB_ID/ray" "$ALIAS"
mkdir -p "$ALIAS/t"
export RAY_TMPDIR="$ALIAS" TMPDIR="$ALIAS/t" XDG_CACHE_HOME="$RUN/cache"
export TRITON_CACHE_DIR="$RUN/cache/triton" TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions"
export HF_HOME="$ROOT/models/huggingface"
cleanup() {
  result=$?
  if [[ "$(readlink "$ALIAS" || true)" == "$RUN/runtime/job$SLURM_JOB_ID/ray" ]]; then unlink "$ALIAS"; fi
  if (( result != 0 )); then printf '{"job_id":"%s","exit_code":%d}\n' "$SLURM_JOB_ID" "$result" > "$RUN/FAILED-$SLURM_JOB_ID.json"; fi
}
trap cleanup EXIT
ulimit -c 0
python - <<'PY'
import torch
assert torch.cuda.device_count() == 2
assert all('A100' in torch.cuda.get_device_properties(i).name and
           39 * 1024**3 < torch.cuda.get_device_properties(i).total_memory < 42 * 1024**3 for i in range(2))
from experiments.joint_config import load, validate
c=load('configs/short8-profile.json'); validate(c, training=True)
assert c['retrieval']['steps']==2 and c['short_episode']['blocks']==8
assert c['experiment']['train_roles']==['B','U','R']
print('A100X2_SHORT8_PROFILE_VERIFIED')
PY
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
DATA="$ROOT/workspace/MemT-Joint/artifacts/short8-r3-20261113/data/train"
python cli.py preflight --config configs/short8-profile.json --data "$DATA" > "$RUN/preflight.json"
# Two REAL updates, no full validation; profile weights are never resumed into the formal 50.
python -u cli.py train --config configs/short8-profile.json --variant M4 \
  --data "$DATA" --output "$RUN/train" > "$RUN/train.log" 2>&1
python -m scripts.verify_short_profile --run "$RUN" \
  --output "$ROOT/workspace/MemT-Joint/artifacts/short8-r3-20261113/PROFILE_PASSED.json"
