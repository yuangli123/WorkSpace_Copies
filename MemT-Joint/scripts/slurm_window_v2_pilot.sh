#!/bin/bash
#SBATCH --job-name=memtj-window3
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@1800
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
export TMPDIR="$ROOT/t$SLURM_JOB_ID" RAY_TMPDIR="$ROOT/r$SLURM_JOB_ID"
RUN="$ROOT/workspace/AgentMemory/artifacts/memt-joint/window-v2-pilot-job$SLURM_JOB_ID"
[[ ! -e "$RUN" ]]
mkdir -p "$RUN" "$TMPDIR" "$RAY_TMPDIR"
export XDG_CACHE_HOME="$RUN/cache" TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions" HF_HOME="$ROOT/models/huggingface"
ulimit -c 0
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python -m scripts.check
BUNDLE="${WINDOW_BUNDLE:?reviewed window bundle required}"
python - "$BUNDLE" "$ROOT" <<'VERIFY'
import json,sys,os
from pathlib import Path
p,root=Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve()
assert root in p.parents and p.is_file() and p.stat().st_uid==os.getuid()
assert len(os.environ['TMPDIR'])+37<107
import torch
assert torch.cuda.device_count()==2
assert all('A100' in torch.cuda.get_device_name(i) for i in range(2))
from experiments.joint_config import load,validate
from experiments.joint import corpus
from experiments.window import load_bundle
cfg=load('configs/window-a100x2-pilot.json');validate(cfg,training=True)
episodes,_=corpus(root/'workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full/train')
windows,questions,identity=load_bundle(p,episodes,root/'workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full/train')
assert len(windows)==3 and len({w['eval_session'] for w in windows})==3
print('WINDOW_QA_SNAPSHOT_VERIFIED',len(questions),len(identity),flush=True)
VERIFY
python - "$RUN/run-config.json" "$BUNDLE" "${WINDOW_BASELINE_SOURCE:-}" <<'CONFIG'
import json,sys
from pathlib import Path
cfg={'extends':[str(Path('configs/window-a100x2-pilot.json').resolve())], 'window':{'bundle':str(Path(sys.argv[2]).resolve())}}
if sys.argv[3]:
    cfg['window']['baseline_source']=str(Path(sys.argv[3]).resolve())
Path(sys.argv[1]).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
CONFIG
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full"
exec python -u cli.py train --config "$RUN/run-config.json" --variant M4 \
 --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUN/train"
