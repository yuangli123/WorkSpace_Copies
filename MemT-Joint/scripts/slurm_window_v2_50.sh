#!/bin/bash
#SBATCH --job-name=memtj-window50
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@3600
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/MemT-Joint
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-%j.err
set -euo pipefail
echo 'DISABLED: window-prefix runs are incompatible with memt-common-v1; use a new empty-state run.' >&2
exit 2
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
RUN="$ROOT/workspace/AgentMemory/artifacts/memt-joint/window-v2-50-job$SLURM_JOB_ID"
[[ ! -e "$RUN" ]]
mkdir -p "$RUN"
python scripts/cleanup_root_temp.py "$RUN/root-temp-cleanup.json"
[[ -L "$ROOT/.m" && "$(readlink -f "$ROOT/.m")" == "$PWD/artifacts/tmp" ]]
export TMPDIR="$ROOT/.m/t$SLURM_JOB_ID" RAY_TMPDIR="$ROOT/.m/r$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
CACHE="$PWD/artifacts/tmp/cache-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$CACHE" TRITON_CACHE_DIR="$CACHE/triton" TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions" HF_HOME="$ROOT/models/huggingface"
ulimit -c 0
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python -m scripts.check
SOURCE="${RESUME_SOURCE:?sealed step1 source required}"
python - "$RUN/run-config.json" "$SOURCE" <<'CONFIG'
import json,sys,os,shutil
from pathlib import Path
root=Path('/public/home/wwxia/yuang.li').resolve();source=Path(sys.argv[2]).resolve()
assert root in source.parents and source.stat().st_uid==os.getuid()
assert shutil.disk_usage(root).free>160*1024**3
assert len(os.environ['TMPDIR'].encode())+37<107
import torch
assert torch.cuda.device_count()==2 and all('A100' in torch.cuda.get_device_name(i) for i in range(2))
old=json.loads((source/'manifest.json').read_text())
cfg={'extends':[str(Path('configs/window-a100x2-50.json').resolve())],
     'window':{'bundle':old['config']['window']['bundle']}}
Path(sys.argv[1]).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
CONFIG
python -u scripts/fork_window50.py "$RUN/run-config.json" "$SOURCE" "$RUN/train"
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full"
exec python -u cli.py train --config "$RUN/run-config.json" --variant M4 \
 --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUN/train" --resume
