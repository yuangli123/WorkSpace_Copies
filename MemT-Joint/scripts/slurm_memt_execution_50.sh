#!/bin/bash
#SBATCH --job-name=memtj-v2-m4-50
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@64800
#SBATCH --no-requeue
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/MemT-Joint-execution-v2-train50-r2-20261113
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/execution-v2-train50-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/execution-v2-train50-%j.err
set -euo pipefail
ROOT=/public/home/wwxia/yuang.li
PROJECT="$ROOT/workspace/MemT-Joint-execution-v2-train50-r2-20261113"
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
RUN="${TRAIN_RUN:-$ROOT/workspace/MemT-Joint/artifacts/execution-v2-train50-job$SLURM_JOB_ID}"
[[ "$RUN" =~ ^/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/execution-v2-train50-job[0-9]+$ ]]
if [[ -z "${TRAIN_RUN:-}" ]]; then [[ ! -e "$RUN" ]]; fi
mkdir -p "$RUN/runtime/job$SLURM_JOB_ID/ray" "$RUN/cache" "$RUN/continuations"
ALIAS="$ROOT/workspace/.k"
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
TRAIN_PID=''
walltime_stop() {
  touch "$RUN/runtime/job$SLURM_JOB_ID/WALLTIME_STOP"
  if [[ -n "$TRAIN_PID" ]]; then kill -USR1 "$TRAIN_PID"; fi
}
trap walltime_stop USR1
ulimit -c 0
python - <<'PY'
import tempfile, uuid, zmq, torch
from pathlib import Path
path = Path(tempfile.gettempdir()) / str(uuid.uuid4())
assert len(str(path).encode()) <= zmq.IPC_PATH_MAX_LEN
ctx = zmq.Context(); sock = ctx.socket(zmq.PUB)
sock.bind('ipc://' + str(path)); sock.close(); ctx.term()
assert torch.cuda.device_count() == 2
assert all('A100' in torch.cuda.get_device_properties(i).name and
           39 * 1024**3 < torch.cuda.get_device_properties(i).total_memory < 42 * 1024**3 for i in range(2))
print('A100X2_IPC_VERIFIED')
PY
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
from experiments.joint_config import load, validate, contract
root = Path(sys.argv[1])
config = load('configs/memt-execution-m4-50.json'); validate(config, training=True)
assert config['retrieval']['steps'] == 50 and config['experiment']['train_roles'] == ['B','U','R']
assert config['runtime']['save_every'] == config['runtime']['validate_every'] == 10
assert config['joint']['builder_rollouts'] > 0 and config['joint']['updater_rollouts'] > 0
assert config['runtime']['checkpoint_retention'] == 'latest_best'
baseline = root/'workspace/MemT-Joint/artifacts/execution-v2-step0-job253511'
assert json.loads((baseline/'STEP0_COMPLETE.json').read_text())['count'] == 81
assert json.loads((baseline/'eval/manifest.json').read_text())['common_protocol'] == contract(config)
capacity = json.loads((root/'workspace/AgentMemory/artifacts/memt-joint/common40k-capacity-job250118/CAPACITY_PASSED.json').read_text())
assert capacity['passed'] and capacity['capacity_only'] and capacity['formal_updates'] == 0
assert {'full_length_update_passed','warm_optimizer_update_passed'} <= {s['name'] for s in capacity['stages']}
print('PROTOCOL_CAPACITY_VERIFIED')
PY
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full"
python cli.py preflight --config configs/memt-execution-m4-50.json --data "$DATA/train" > "$RUN/preflight-$SLURM_JOB_ID.json"
set --
if [[ -f "$RUN/train/LATEST.json" ]]; then set -- --resume; fi
python -u cli.py train --config configs/memt-execution-m4-50.json --variant M4 \
  --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUN/train" "$@" \
  > "$RUN/train-$SLURM_JOB_ID.log" 2>&1 &
TRAIN_PID=$!
while true; do
  if wait "$TRAIN_PID"; then result=0; break; else result=$?; fi
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then break; fi
done
TRAIN_PID=''
(( result == 0 )) || exit "$result"
if [[ -f "$RUN/train/TRAINING_COMPLETE.json" ]]; then
  python - "$RUN/train" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); state=json.loads((p/'TRAINING_COMPLETE.json').read_text())
assert state['step']==50
latest=json.loads((p/'LATEST.json').read_text())
receipt=json.loads((p/'validation'/(Path(latest['path']).name+'.json')).read_text())
summary=json.loads((p/receipt['validation']/'summary.json').read_text())
assert summary['count']==81
updates=[]
for file in p.glob('attempts/*/step-*/metrics.json'):
    metrics=json.loads(file.read_text())
    step=int(file.parent.name.split('-')[1])
    assert metrics['iteration']+1==step
    updates.append({'step':step,'metrics':metrics,'source':str(file.relative_to(p))})
assert sorted(u['step'] for u in updates)==list(range(1,51))
validations=[]
for file in p.glob('validation/step-*.json'):
    selected=json.loads(file.read_text())
    result=json.loads((p/selected['validation']/'summary.json').read_text())
    assert result['count']==81
    validations.append({'step':selected['step'],'summary':result,'source':selected['validation']})
assert sorted(v['step'] for v in validations)==[0,10,20,30,40,50]
(p.parent/'TRAJECTORY_0_50.json').write_text(json.dumps({
    'updates':sorted(updates,key=lambda x:x['step']),
    'validations':sorted(validations,key=lambda x:x['step']),
    'optimizer_updates':50,'protocol':'memt-execution-v2'},ensure_ascii=False,indent=2))
print('TRAINING_AND_FINAL_VALIDATION_COMPLETE',summary['f1'],summary['bleu1'])
PY
elif [[ -f "$RUN/runtime/job$SLURM_JOB_ID/WALLTIME_STOP" && -f "$RUN/train/STOPPED.json" && ! -e "$RUN/train/STOP" ]]; then
  # Continue this finite 50-update run only after a clean wall-time checkpoint.
  NEXT=$(sbatch --parsable --dependency="afterok:$SLURM_JOB_ID" --export="ALL,TRAIN_RUN=$RUN" "$PROJECT/scripts/slurm_memt_execution_50.sh")
  printf '{"previous_job":"%s","next_job":"%s","run":"%s"}\n' "$SLURM_JOB_ID" "$NEXT" "$RUN" > "$RUN/continuations/job-$SLURM_JOB_ID.json"
  echo "CONTINUATION_SUBMITTED $NEXT"
fi
