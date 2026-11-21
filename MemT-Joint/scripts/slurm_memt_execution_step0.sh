#!/bin/bash
#SBATCH --job-name=memtj-v2-step0
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/MemT-Joint-execution-v2-r2-20261112
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/execution-v2-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/execution-v2-%j.err
set -euo pipefail
ROOT=/public/home/wwxia/yuang.li
PROJECT="$ROOT/workspace/MemT-Joint-execution-v2-r2-20261112"
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
RUN="$ROOT/workspace/MemT-Joint/artifacts/execution-v2-step0-job$SLURM_JOB_ID"
[[ ! -e "$RUN" ]]
mkdir -p "$RUN/runtime/tmp" "$RUN/runtime/ray" "$RUN/cache"
export TMPDIR="$RUN/runtime/tmp" XDG_CACHE_HOME="$RUN/cache"
export TRITON_CACHE_DIR="$RUN/cache/triton" TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions"
export HF_HOME="$ROOT/models/huggingface"
# Ray Unix sockets need a short alias; all actual files stay in project artifacts.
ALIAS="$ROOT/workspace/.j"
[[ ! -e "$ALIAS" && ! -L "$ALIAS" ]]
ln -s "$RUN/runtime/ray" "$ALIAS"
export RAY_TMPDIR="$ALIAS"
mkdir -p "$ALIAS/t"
export TMPDIR="$ALIAS/t"
cleanup() {
  status=$?
  if [[ "$(readlink "$ALIAS" || true)" == "$RUN/runtime/ray" ]]; then unlink "$ALIAS"; fi
  if (( status != 0 )); then printf '{"exit_code":%d}\n' "$status" > "$RUN/FAILED.json"; fi
}
trap cleanup EXIT
ulimit -c 0
python - <<'PY'
import tempfile, uuid, zmq
from pathlib import Path
path = Path(tempfile.gettempdir()) / str(uuid.uuid4())
assert len(str(path).encode()) <= zmq.IPC_PATH_MAX_LEN
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('ipc://' + str(path))
socket.close()
context.term()
print('IPC_PATH_VERIFIED', str(path))
PY
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python - <<'PY'
import torch
assert torch.cuda.device_count() == 2
for index in range(2):
    p = torch.cuda.get_device_properties(index)
    assert 'A100' in p.name and 39 * 1024**3 < p.total_memory < 42 * 1024**3
PY
python -m scripts.verify_memt_backend --model "$ROOT/models/bge-m3" --output "$RUN/backend" > "$RUN/backend.log" 2>&1
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full/valid"
python cli.py preflight --config configs/memt-execution-step0.json --data "$DATA" > "$RUN/preflight.json"
python -u cli.py evaluate --config configs/memt-execution-step0.json --data "$DATA" --variant M0 \
  --output "$RUN/eval" > "$RUN/evaluate.log" 2>&1
python - "$RUN" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
summary = json.loads((run/'eval/summary.json').read_text())
rows = [json.loads(line) for line in (run/'eval/answers.jsonl').read_text().splitlines()]
assert summary['count'] == len(rows) == len({r['qa_id'] for r in rows}) == 81
assert summary['common_protocol'] == 'memt-execution-v2'
assert summary['answer_weights'] == 'current_policy'
assert summary['protocol_contract']['baseline_reuse'] is False
assert summary['protocol_contract']['role_tags'] is False
(run/'STEP0_COMPLETE.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False))
PY
