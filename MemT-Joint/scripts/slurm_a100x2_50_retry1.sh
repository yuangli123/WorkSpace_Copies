#!/bin/bash
#SBATCH --job-name=memtj-50x2-r1
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
echo 'DISABLED: baseline-reuse retry is incompatible with memt-common-v1; use a new empty-state run.' >&2
exit 2
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
export TMPDIR="$PWD/artifacts/tmp-$SLURM_JOB_ID" RAY_TMPDIR="$ROOT/r$SLURM_JOB_ID"
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full"
RUN="$ROOT/workspace/AgentMemory/artifacts/memt-joint/a100x2-50-20261106-job$SLURM_JOB_ID"
BASELINE="$ROOT/workspace/AgentMemory/artifacts/memt-joint/a100x2-50-20261106-job250022/step000-eval"
[[ ! -e "$RUN" ]]
mkdir -p "$RUN" "$TMPDIR" "$RAY_TMPDIR"
export XDG_CACHE_HOME="$RUN/cache" TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCH_EXTENSIONS_DIR="$RUN/cache/torch_extensions" HF_HOME="$ROOT/models/huggingface"
ulimit -c 0
python -c 'import torch; assert torch.cuda.device_count()==2; print("ALLOCATED_GPU_COUNT", torch.cuda.device_count(), flush=True)'
nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python -m pip check
python -m scripts.check
# Check the actual distributed import chain before loading any training weights.
python -c 'from training.verl_backend import enable_verl; enable_verl(); from verl.trainer.ppo.ray_trainer_ts import RayPPOTrainer; from verl.workers.fsdp_workers import ActorRolloutRefWorker; print("VERL_IMPORT_READY", flush=True)'
python cli.py preflight --config configs/a100x2-50.json --data "$DATA/train" > "$RUN/preflight.json"
python - "$BASELINE" "$RUN" "$DATA/valid" <<'PY'
import os, sys, shutil
from pathlib import Path
from experiments.io import read_json, write_json, sha256
source, target, valid = map(Path, sys.argv[1:])
root = Path('/public/home/wwxia/yuang.li').resolve()
assert all(root in p.resolve().parents and p.stat().st_uid == os.getuid() for p in (source, target, valid))
receipt = read_json(source / 'COMPLETE.json')
assert receipt['answers_sha256'] == sha256(source / 'answers.jsonl') and not receipt['fixture_only']
old, current = read_json(source / 'manifest.json'), read_json(target / 'preflight.json')['identity']
assert old['models'] == current['models']
assert all(sha256(valid / name) == old['inputs'][name] for name in ('episodes.jsonl','questions.jsonl','partition.json'))
assert old['config']['experiment']['variant'] == 'M0'
assert old['config']['experiment']['mode'] == current['config']['experiment']['mode']
for key in ('model','memory','evaluation'):
    assert old['config'][key] == current['config'][key], key
changed = {name for name in set(old['code']) | set(current['code']) if old['code'].get(name) != current['code'].get(name)}
assert changed <= {'experiments/joint.py', 'third_party/Tree-GRPO/verl/version/version'}, changed
shutil.copytree(source, target / 'step000-eval')
write_json(target / 'BASELINE_REUSED.json', {'source':str(source), 'answers_sha256':receipt['answers_sha256'],
           'manifest_sha256':sha256(source/'manifest.json'), 'summary_sha256':sha256(source/'summary.json'),
           'code_changes_for_runtime_asset_repair':sorted(changed)})
print('VERIFIED_BASELINE_REUSED', read_json(source / 'summary.json'), flush=True)
PY
exec python -u cli.py train --config configs/a100x2-50.json --variant M4 \
  --data "$DATA/train" --valid-data "$DATA/valid" --output "$RUN/train"
