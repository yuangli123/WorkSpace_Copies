#!/bin/bash
# Evaluate the completed step-50 actor with the unified current-policy Answer path.
# This job evaluates the completed export only; it never calls cli.py train.
#SBATCH --job-name=memtj-eval50
#SBATCH --partition=exclusive_4
#SBATCH --account=wwxia
#SBATCH --qos=user_wwxia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/public/home/wwxia/yuang.li/workspace/MemT-Joint-eval-nonfrozen-20261110
#SBATCH --output=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-eval-%j.out
#SBATCH --error=/public/home/wwxia/yuang.li/workspace/MemT-Joint/artifacts/slurm-eval-%j.err
set -euo pipefail

ROOT=/public/home/wwxia/yuang.li
PROJECT="$ROOT/workspace/MemT-Joint-eval-nonfrozen-20261110"
[[ "$PWD" == "$PROJECT" && "$(stat -c %u "$ROOT")" == "$UID" ]]
set +u
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/miniforge3/envs/memt-joint"
set -u
[[ "$CONDA_PREFIX" == "$ROOT/miniforge3/envs/memt-joint" ]]

export PYTHONPATH="$PROJECT:$PROJECT/third_party/Tree-GRPO"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
export RAY_enable_worker_prestart=0 RAY_prestart_worker_first_driver=0
export TMPDIR="$ROOT/workspace/AgentMemory/artifacts/t$SLURM_JOB_ID"
export RAY_TMPDIR="$ROOT/reval-$SLURM_JOB_ID"
export XDG_CACHE_HOME="$PROJECT/artifacts/cache-eval-$SLURM_JOB_ID"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
export HF_HOME="$ROOT/models/huggingface"
ulimit -c 0

RUN="$ROOT/workspace/AgentMemory/artifacts/memt-joint/nonfrozen-eval-step50-job$SLURM_JOB_ID"
DATA="$ROOT/workspace/Mem-T/artifacts/retrieval-trial-20261026/data/full"
MODEL="$ROOT/workspace/AgentMemory/artifacts/memt-joint/nonfrozen-eval-step50-job251794/model-step50"
[[ ! -e "$RUN" ]]
mkdir -p "$RUN" "$TMPDIR" "$RAY_TMPDIR" "$XDG_CACHE_HOME"

nvidia-smi --id "$CUDA_VISIBLE_DEVICES"
python -m py_compile experiments/joint.py experiments/joint_config.py methods/joint/prompts.py \
  methods/joint/retrieval.py methods/joint/credit.py methods/joint/rollout.py models/joint.py memory/branch.py

# Reuse only actor weights, never an old memory snapshot or baseline result.
python - "$MODEL/COMPLETE.json" <<'PY'
import json, sys
from pathlib import Path
assert json.loads(Path(sys.argv[1]).read_text())['step'] == 50
PY

python cli.py preflight --config configs/common-a100x2-50.json --data "$DATA/valid" > "$RUN/preflight.json"
python -u cli.py evaluate --config configs/common-a100x2-50.json --data "$DATA/valid" \
  --model "$MODEL" --output "$RUN/eval" > "$RUN/evaluate.jsonl"

python - "$RUN/eval/summary.json" "$RUN/eval/answers.jsonl" <<'PY'
import json, sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text())
assert summary['count'] == 81
assert summary['answer_weights'] == 'current_policy'
assert summary['common_protocol'] == 'memt-execution-v2'
rows = [json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines() if line.strip()]
assert len(rows) == 81
Path(Path(sys.argv[1]).parent.parent / 'NONFROZEN_EVAL_COMPLETE.json').write_text(json.dumps({
    'count': summary['count'], 'f1': summary['f1'], 'bleu1': summary['bleu1'],
    'strict_f1': summary['strict_f1'], 'strict_bleu1': summary['strict_bleu1'],
    'finish_rate': summary['finish_rate'], 'fallback_rate': summary['fallback_rate'],
    'answer_weights': summary['answer_weights'], 'protocol': summary['common_protocol'],
}, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False), flush=True)
PY
