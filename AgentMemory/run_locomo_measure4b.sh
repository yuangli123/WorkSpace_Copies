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
bench=${BENCH:-locomo}
selection=()
case "$bench" in
  locomo)
    scope=full-4b-measured-20261003
    data_path=$LOCOMO_PATH
    expected_sha=79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
    ;;
  longmemeval_s)
    scope=full-4b-measured-20261004
    data_path=$LONGMEMEVAL_S_PATH
    expected_sha=d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
    ;;
  *) echo "Unsupported BENCH: $bench" >&2; exit 2 ;;
esac
if [[ "${1-}" == --smoke ]]; then
  if [[ "$bench" == locomo ]]; then
    scope=smoke-4b-measured-20261003
    data_path="$A/artifacts/data/smoke/locomo_one_session_two_questions.json"
    expected_sha=cc25b8381b2981b78cd83d05f017cf808fb95c9596c774cc3d964d175c5451d1
  else
    scope=smoke-4b-measured-20261004
  fi
  selection=(--conv_limit 1)
elif [[ $# -ne 0 ]]; then
  echo "Usage: $0 [--smoke]" >&2
  exit 2
fi
R="$A/artifacts/results/comparison/$bench/$scope"
S="$A/artifacts/stores/comparison/$bench/$scope"
L="$A/artifacts/logs/$scope"
P=${bench}_${scope//-/_}
if [[ "$bench" == locomo ]]; then
  export LOCOMO_PATH="$data_path" LOCOMO_SCORE_PATH="$R/scores"
else
  export LONGMEMEVAL_S_PATH="$data_path" LONGMEMEVAL_S_SCORE_PATH="$R/scores"
fi
mkdir -p "$R/scores" "$S" "$L"
exec 9>"$R/driver.lock"
flock -n 9 || { echo "Measured LoCoMo driver is already running" >&2; exit 1; }

test "$(sha256sum "$data_path" | cut -d' ' -f1)" = "$expected_sha"
test "$(curl -fsS http://127.0.0.1:8000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = qwen3-4b-chat
test "$(curl -fsS http://127.0.0.1:8001/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = qwen3-embedding-0.6b

conda activate /public/home/wwxia/yuang.li/miniforge3/envs/recmem
for method in fullcontext naive_rag recmem; do
  if [[ -s "$R/scores/${P}_${method}.json.run.json" ]]; then
    echo "$(date -Is) $method already scored; skipping"
    continue
  fi
  echo "$(date -Is) $method start"
  extra=()
  top_k=10
  if [[ "$method" == naive_rag ]]; then
    top_k=3
    extra+=(--rag_store "$S/naive_rag" --rag_tokenizer /public/home/wwxia/yuang.li/models/Qwen3-4B-Instruct-2507/tokenizer.json)
  elif [[ "$method" == recmem ]]; then
    extra+=(--semantic_store "$S/recmem/semantic" --subconscious_store "$S/recmem/subconscious" --episodic_store "$S/recmem/episodic")
  fi
  python -m evaluation.run_experiments --method "$method" --bench "$bench" \
    --model qwen3-4b-chat --judge_model qwen3-4b-chat --resume --measure \
    --output_file "$R/${P}_${method}.json" --top_k "$top_k" --context_budget 12000 \
    --conv_workers 1 --question_max_workers 1 "${selection[@]}" "${extra[@]}" > "$L/$method.log" 2>&1
  echo "$(date -Is) $method complete"
done

for method in memoryos mem0 amem; do
  if [[ -s "$R/scores/${P}_${method}.json.run.json" ]]; then
    echo "$(date -Is) $method already scored; skipping"
    continue
  fi
  echo "$(date -Is) $method start"
  conda activate "/public/home/wwxia/yuang.li/miniforge3/envs/recmem-$method"
  python -m evaluation.external_runner --method "$method" --bench "$bench" \
    --data_path "$data_path" --output_file "$R/${P}_${method}.json" \
    --store_root "$S/$method" --model qwen3-4b-chat \
    --embedding_model qwen3-embedding-0.6b --embedding_dim 1024 \
    --top_k 10 --context_budget 12000 --resume --measure "${selection[@]}" > "$L/$method.log" 2>&1
  conda activate /public/home/wwxia/yuang.li/miniforge3/envs/recmem
  python -m evaluation.run_experiments --bench "$bench" --eval_only --measure \
    --output_file "$R/${P}_${method}.json" --judge_model qwen3-4b-chat \
    > "$L/$method-score.log" 2>&1
  echo "$(date -Is) $method complete"
done

runs=()
for method in recmem fullcontext naive_rag memoryos mem0 amem; do
  runs+=(--run "$method=$R/${P}_${method}.json=$R/scores/${P}_${method}.json")
done
if [[ ! -e "$R/comparison.json" ]]; then
  python -m evaluation.compare_results --bench "$bench" --data_path "$data_path" \
    "${selection[@]}" "${runs[@]}" --metrics --output "$R/comparison.json"
fi
echo "$(date -Is) comparison complete"
