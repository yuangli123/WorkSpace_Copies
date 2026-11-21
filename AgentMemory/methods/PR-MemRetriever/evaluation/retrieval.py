"""Greedy tool-using retrieval evaluation and answer metrics."""
import copy
import random
from pathlib import Path
from time import perf_counter
from experiments.io import read_json, sha256, write_json, write_jsonl
from methods.retrieval.tree import sample_tree, answer_f1, answer_bleu1

def locomo_abstention(answer):
    lowered = answer.lower()
    return "no information available" in lowered or "not mentioned" in lowered


def metric_summary(rows):
    count = len(rows)
    latencies = sorted(row["latency_seconds"] for row in rows)
    scored = [row for row in rows if row["f1"] is not None]
    abstentions = [row["abstention_correct"] for row in rows if "abstention_correct" in row]

    def mean(field):
        return sum(row[field] for row in rows) / max(1, count)

    def percentile(fraction):
        if not latencies:
            return None
        index = fraction * (len(latencies) - 1)
        lower = int(index)
        return latencies[lower] + (index - lower) * (latencies[min(lower + 1, len(latencies) - 1)] - latencies[lower])

    decisions = sum(row["steps"] for row in rows)
    searches = sum(row["search_rounds"] for row in rows)
    scores = {name: sum(row[name] for row in scored) / max(1, len(scored)) for name in ("f1", "bleu1")}
    if abstentions:
        scores.update(f1_count=len(scored), abstention_count=len(abstentions),
                      abstention_accuracy=sum(abstentions)/len(abstentions))
    return {"count": count, **scores,
            "finish_rate": mean("finished"), "fallback_rate": mean("fallback_answer"),
            "format_success_rate": 1 - sum(row["format_errors"] for row in rows) / decisions if decisions else None,
            "search_budget_errors": sum(row["search_budget_errors"] for row in rows),
            "parallel_round_rate": sum(row["parallel_rounds"] for row in rows) / searches if searches else None,
            "mean_decisions": mean("steps"), "mean_search_rounds": mean("search_rounds"),
            "mean_search_calls": mean("search_calls"), "mean_generated_tokens": mean("generated_tokens"),
            "mean_prompt_tokens": mean("prompt_tokens"),
            "mean_policy_seconds": mean("policy_seconds"), "mean_tool_seconds": mean("tool_seconds"),
            "latency_p50_seconds": percentile(0.50), "latency_p95_seconds": percentile(0.95)}


def evaluate(config, memory, policy, output, limit=None):
    rows = []
    inference = copy.deepcopy(config)
    inference["retrieval"]["max_depth"] = config["evaluation"]["max_steps"]
    inference["retrieval"].update(
        expansion_rounds=0, measure_evaluation=True,
        max_parallel_calls=config["evaluation"].get("max_parallel_calls", 1),
        serial_tool_execution=config["evaluation"].get("serial_tool_execution", False),
    )
    if "max_search_calls" in config["evaluation"]:
        inference["retrieval"]["max_search_calls"] = config["evaluation"]["max_search_calls"]
    limit = config["evaluation"].get("limit") if limit is None else limit
    class Greedy:
        def complete(self, messages, **kwargs):
            return policy.complete(messages, temperature=0, **kwargs)
        def check_observation(self, text):
            return policy.check_observation(text)
    inference_policy = Greedy()
    managed = config.get('runtime', {}).get('resumable_evaluation', False)
    for index, qa in enumerate(memory.questions[:limit]):
        saved = Path(output).parent / 'questions' / f'{index:06d}.json.gz'
        seal = saved.with_suffix('.sha256.json')
        if managed and saved.exists():
            if not seal.exists() or read_json(seal)['sha256'] != sha256(saved):
                raise ValueError('Evaluation row seal missing or changed')
            row = read_json(saved)
            if any(row[k] != qa[k] for k in ('sample_id', 'qa_id', 'question', 'answer', 'category')):
                raise ValueError('Evaluation row does not match ordered questions')
            rows.append(row)
            continue
        if managed and hasattr(policy, 'set_seed'):
            policy.set_seed(config['seed'] + index)
        env = memory.environment(qa, inference, Path(output).parent / "tool_logs")
        started = perf_counter()
        nodes = sample_tree(inference_policy, env, qa, inference, random.Random(config["seed"]))
        leaf = nodes[-1]
        answer, fallback, fallback_seconds = leaf.answer, False, 0.0
        extra_prompt, extra_response = 0, 0
        if not (leaf.valid and leaf.terminal) and config["evaluation"].get("fallback_answer", True):
            from memory.retrieval import AnswerWithMemoriesTool
            class AnswerPolicy:
                def get_completion(self, text):
                    nonlocal extra_prompt, extra_response
                    response = policy.complete([{"role": "user", "content": text}], temperature=0, stop_at_tool=False)
                    extra_prompt, extra_response = len(response.prompt_ids), len(response.response_ids)
                    return response.text
            answer_tool = AnswerWithMemoriesTool(AnswerPolicy(), config.get("benchmark", "locomo"), str(qa["category"]))
            fallback_started = perf_counter()
            answer, _ = answer_tool(qa["question"], "\n\n".join(n.observation for n in nodes[1:] if n.observation))
            fallback = True
            fallback_seconds = perf_counter() - fallback_started
        latency = perf_counter() - started
        stats = [node.evaluation or {} for node in nodes[1:]]
        calls = [stat.get("search_calls", 0) for stat in stats]
        row = {"qa_id": qa["qa_id"], "sample_id": qa["sample_id"], "category": qa["category"],
                     "question": qa["question"], "answer": qa["answer"], "response": answer,
                     "gold": qa["answer"], "pred": answer,
                     "f1": answer_f1(answer, qa["answer"]), "bleu1": answer_bleu1(answer, qa["answer"]),
                     "fallback_answer": fallback,
                     "finished": leaf.valid and leaf.terminal, "steps": leaf.depth,
                     "latency_seconds": latency,
                     "policy_seconds": sum(stat.get("decision_seconds", 0) for stat in stats) + fallback_seconds,
                     "tool_seconds": sum(stat.get("tool_seconds", 0) for stat in stats),
                     "search_calls": sum(calls), "search_rounds": sum(n > 0 for n in calls),
                     "parallel_rounds": sum(n > 1 for n in calls),
                     "format_errors": sum(stat.get("error") == "format" for stat in stats),
                     "search_budget_errors": sum(stat.get("error") == "search_budget" for stat in stats),
                     "generated_tokens": sum(len(n.response_ids) for n in nodes[1:]) + extra_response,
                     "prompt_tokens": sum(len(n.prompt_ids) for n in nodes[1:]) + extra_prompt,
                     "traces": [{"step": n.depth, "text": n.text, "observation": n.observation,
                                 "retrieved": n.retrieved, **(n.evaluation or {})} for n in nodes[1:]]}
        if qa.get("evaluation_metric") == "locomo_abstention":
            row["f1"] = row["bleu1"] = None
            row["abstention_correct"] = locomo_abstention(answer)
        rows.append(row)
        if managed:
            write_json(saved, row)
            write_json(seal, {'sha256': sha256(saved)})
            write_json(Path(output).parent / 'PROGRESS.json', {'completed': len(rows), 'target': len(memory.questions[:limit])})
    write_jsonl(output, rows)
    summary = metric_summary(rows)
    scores_by_category = {}
    for row in rows:
        if row["f1"] is not None:
            scores_by_category.setdefault(str(row["category"]), []).append(row["f1"])
    summary["by_category"] = {category: sum(scores) / len(scores)
                              for category, scores in sorted(scores_by_category.items())}
    summary_path = Path(output).with_suffix('') if str(output).endswith('.gz') else Path(output)
    write_json(summary_path.with_suffix('.metrics.json'), summary)
    return summary


