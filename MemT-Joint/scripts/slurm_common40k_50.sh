#!/bin/bash
#SBATCH --job-name=memtj-common50
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
set +u
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/miniforge3/envs/memt-joint"
set -u
[[ "$CONDA_PREFIX" == "$ROOT/miniforge3/envs/memt-joint" ]]
export PYTHONPATH="$PWD:$PWD/third_party/Tree-GRPO"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0
export TMPDIR="$PWD/artifacts/tmp-$SLURM_JOB_ID" RAY_TMPDIR="$ROOT/r$SLURM_JOB_ID"
RUN="$ROOT/workspace/AgentMemory/artifacts/memt-joint/common40k-50-job$SLURM_JOB_ID"
[[ ! -e "$RUN" ]]
mkdir -p "$RUN" "$TMPDIR" "$RAY_TMPDIR"
export XDG_CACHE_HOME="$RUN/cache" TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions" HF_HOME="$ROOT/models/huggingface"
ulimit -c 0
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python -m scripts.check
CAPACITY="$ROOT/workspace/AgentMemory/artifacts/memt-joint/common40k-capacity-job${CAPACITY_JOB_ID:?capacity job required}"
python - "$CAPACITY" <<'VERIFY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'CAPACITY_PASSED.json').read_text())
assert r['passed'] is True and r['capacity_only'] and r['formal_updates']==0
stages={s['name']:s for s in r['stages']}
assert stages['vllm_prefill']['prompt_tokens']>=39928
assert stages['full_length_update']['response_tokens']==1024
assert 'full_length_update_passed' in stages
assert 'warm_optimizer_update_passed' in stages
assert r['contract']['prompt_tokens']==39936 and r['contract']['output_tokens']==1024
print('CAPACITY_VERIFIED',sys.argv[1],flush=True)
VERIFY
BRIDGE="$ROOT/workspace/AgentMemory/artifacts/memt-joint/common40k-bridge-job${BRIDGE_JOB_ID:?bridge job required}"
python - "$BRIDGE" <<'VERIFY'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'BRIDGE_PASSED.json').read_text())
assert r['passed'] and r['capacity_only'] and r['formal_updates']==0
assert r['closed_cleanly'] and r['single_engine_role_switch']
print('NATIVE_VLLM_BRIDGE_VERIFIED',flush=True)
VERIFY
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full"
python cli.py preflight --config configs/common-a100x2-50.json --data "$DATA/train" > "$RUN/preflight.json"
exec python -u cli.py train --config configs/common-a100x2-50.json --variant M4 \
 --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUN/train"
