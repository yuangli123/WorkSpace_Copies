#!/usr/bin/env bash
set -eo pipefail

A=/public/home/wwxia/yuang.li/workspace/AgentMemory
source /public/home/wwxia/yuang.li/miniforge3/etc/profile.d/conda.sh
cd "$A/methods/RecMem"
bash -n "$A/.env"
set -a
source "$A/.env"
set +a
export PYTHONPATH="$A:$A/methods/RecMem"

R="$A/artifacts/results/comparison/longmemeval_s/full-4b-measured-20261004"
S="$A/artifacts/stores/comparison/longmemeval_s/full-4b-measured-20261004"
L="$A/artifacts/logs/full-4b-measured-20261004"
P=longmemeval_s_full_4b_measured_20261004
DATA="$LONGMEMEVAL_S_PATH"
export LONGMEMEVAL_S_SCORE_PATH="$R/scores"

test "$(sha256sum "$DATA" | cut -d' ' -f1)" = d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
test "$(curl -fsS http://127.0.0.1:8000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = qwen3-4b-chat
test "$(curl -fsS http://127.0.0.1:8001/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = qwen3-embedding-0.6b
exec 9>"$R/driver.lock"
flock -n 9 || { echo "Another comparison driver is running" >&2; exit 1; }

for method in mem0 amem; do
  if [[ -s "$R/scores/${P}_${method}.json.run.json" ]]; then
    echo "$(date -Is) $method already scored; skipping"
    continue
  fi
  echo "$(date -Is) $method start"
  conda activate "/public/home/wwxia/yuang.li/miniforge3/envs/recmem-$method"
  python -m evaluation.external_runner --method "$method" --bench longmemeval_s \
    --data_path "$DATA" --output_file "$R/${P}_${method}.json" \
    --store_root "$S/$method" --model qwen3-4b-chat \
    --embedding_model qwen3-embedding-0.6b --embedding_dim 1024 \
    --top_k 10 --context_budget 12000 --resume --measure >> "$L/$method.log" 2>&1
  conda activate /public/home/wwxia/yuang.li/miniforge3/envs/recmem
  python -m evaluation.run_experiments --bench longmemeval_s --eval_only --measure \
    --output_file "$R/${P}_${method}.json" --judge_model qwen3-4b-chat \
    >> "$L/$method-score.log" 2>&1
  echo "$(date -Is) $method complete"
done
echo "$(date -Is) remaining methods complete; comparison awaits MemoryOS"
