#!/usr/bin/env bash
set -euo pipefail

A=/public/home/wwxia/yuang.li/workspace/AgentMemory
source /public/home/wwxia/yuang.li/miniforge3/etc/profile.d/conda.sh
cd "$A/methods/RecMem"
set -a
source "$A/.env"
set +a
export PYTHONPATH="$A:$A/methods/RecMem"
R="$A/artifacts/results/comparison/locomo/full-4b-20261002"
S="$A/artifacts/stores/comparison/locomo/full-4b-20261002"
L="$A/artifacts/logs/full-4b-20261002"
P=locomo_full4b_20261002
export LOCOMO_SCORE_PATH="$R/scores"
test "$(sha256sum "$LOCOMO_PATH" | cut -d' ' -f1)" = 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
test "$(curl -fsS http://127.0.0.1:8000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = qwen3-4b-chat
test "$(curl -fsS http://127.0.0.1:8001/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = qwen3-embedding-0.6b
for method in fullcontext naive_rag recmem memoryos; do
  test "$(jq length "$R/${P}_${method}.json")" = 10
  test -s "$R/scores/${P}_${method}.json.run.json"
done

for method in mem0 amem; do
  echo "$(date -Is) $method resume start"
  conda activate "/public/home/wwxia/yuang.li/miniforge3/envs/recmem-$method"
  python -m evaluation.external_runner --method "$method" --bench locomo \
    --data_path "$LOCOMO_PATH" --output_file "$R/${P}_${method}.json" \
    --store_root "$S/$method" --model qwen3-4b-chat \
    --embedding_model qwen3-embedding-0.6b --embedding_dim 1024 \
    --top_k 10 --context_budget 12000 --resume > "$L/$method-resume.log" 2>&1
  conda activate /public/home/wwxia/yuang.li/miniforge3/envs/recmem
  python -m evaluation.run_experiments --bench locomo --eval_only \
    --output_file "$R/${P}_${method}.json" --judge_model qwen3-4b-chat \
    > "$L/$method-score.log" 2>&1
  echo "$(date -Is) $method complete"
done

RUNS=()
for method in recmem fullcontext naive_rag memoryos mem0 amem; do
  RUNS+=(--run "$method=$R/${P}_${method}.json=$R/scores/${P}_${method}.json")
done
python -m evaluation.compare_results --bench locomo --data_path "$LOCOMO_PATH" \
  "${RUNS[@]}" --output "$R/comparison.json"
echo "$(date -Is) comparison complete"
