"""Greedy tool-using retrieval evaluation and answer metrics."""
import copy
import random
from pathlib import Path
from experiments.io import read_json, sha256, write_json, write_jsonl
from methods.retrieval.tree import sample_tree, answer_f1, answer_bleu1

def evaluate(config, memory, policy, output, limit=None):
    rows = []
    inference = copy.deepcopy(config)
    inference["retrieval"]["max_depth"] = config["evaluation"]["max_steps"]
    inference["retrieval"]["expansion_rounds"] = 0
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
        nodes = sample_tree(inference_policy, env, qa, inference, random.Random(config["seed"]))
        leaf = nodes[-1]
        answer, fallback, extra_tokens = leaf.answer, False, 0
        if not (leaf.valid and leaf.terminal) and config["evaluation"].get("fallback_answer", True):
            from memory.retrieval import AnswerWithMemoriesTool
            class AnswerPolicy:
                def get_completion(self, text):
                    nonlocal extra_tokens
                    response = policy.complete([{"role": "user", "content": text}], temperature=0, stop_at_tool=False)
                    extra_tokens = len(response.response_ids)
                    return response.text
            answer_tool = AnswerWithMemoriesTool(AnswerPolicy(), config.get("benchmark", "locomo"), str(qa["category"]))
            answer, _ = answer_tool(qa["question"], "\n\n".join(n.observation for n in nodes[1:] if n.observation))
            fallback = True
        row = {"qa_id": qa["qa_id"], "sample_id": qa["sample_id"], "category": qa["category"],
                     "question": qa["question"], "answer": qa["answer"], "response": answer,
                     "gold": qa["answer"], "pred": answer,
                     "f1": answer_f1(answer, qa["answer"]), "bleu1": answer_bleu1(answer, qa["answer"]),
                     "fallback_answer": fallback,
                     "finished": leaf.valid and leaf.terminal, "steps": leaf.depth,
                     "generated_tokens": sum(len(n.response_ids) for n in nodes[1:]) + extra_tokens,
                     "traces": [{"step": n.depth, "text": n.text, "observation": n.observation,
                                 "retrieved": n.retrieved} for n in nodes[1:]]}
        rows.append(row)
        if managed:
            write_json(saved, row)
            write_json(seal, {'sha256': sha256(saved)})
            write_json(Path(output).parent / 'PROGRESS.json', {'completed': len(rows), 'target': len(memory.questions[:limit])})
    write_jsonl(output, rows)
    categories = sorted({str(row["category"]) for row in rows})
    summary = {"count": len(rows), "f1": sum(row["f1"] for row in rows) / max(1, len(rows)),
               "bleu1": sum(row["bleu1"] for row in rows) / max(1, len(rows)),
               "finish_rate": sum(row["finished"] for row in rows) / max(1, len(rows)),
               "by_category": {c: sum(r["f1"] for r in rows if str(r["category"]) == c) /
                                sum(str(r["category"]) == c for r in rows) for c in categories}}
    summary_path = Path(output).with_suffix('') if str(output).endswith('.gz') else Path(output)
    write_json(summary_path.with_suffix('.metrics.json'), summary)
    return summary


