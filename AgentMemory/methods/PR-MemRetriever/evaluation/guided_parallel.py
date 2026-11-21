"""Evaluate constrained multi-round retrieval with matched search-call budgets."""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

from evaluation.retrieval import evaluate
from experiments.io import checked_output, load_config, manifest, read_json, read_jsonl, sha256, write_json
from memory.protocol import BATCH_STOP, parse_batch


def batch_grammar(widths, *, allow_finish):
    """Emit the Mem-T tool-call tags and JSON directly. Each item has its own query."""
    root = [" ws ".join(["search"] * width) for width in widths]
    if allow_finish:
        root.append("finish")
    alternatives = " | ".join(root)
    return (
        'root ::= "<tool_calls>" ws (' + alternatives + ') ws "</tool_calls>"\n'
        r'''
ws ::= [ \t\r\n]*
search ::= "<tool_call>" ws "{" ws "\"name\"" ws ":" ws search_name ws "," ws "\"arguments\"" ws ":" ws "{" ws "\"query\"" ws ":" ws quoted ws "}" ws "}" ws "</tool_call>"
finish ::= "<tool_call>" ws "{" ws "\"name\"" ws ":" ws "\"finish\"" ws "," ws "\"arguments\"" ws ":" ws "{" ws "\"answer\"" ws ":" ws quoted ws "}" ws "}" ws "</tool_call>"
search_name ::= "\"search_facts\"" | "\"search_turns\"" | "\"search_summary\"" | "\"search_personas\"" | "\"search_experiences\""
quoted ::= "\"" ([^"\\<>\x00-\x1f] | "\\" ["\\/bfnrt])+ "\""
'''
    )


class GuidedPolicy:
    def __init__(self, policy, mode, *, width=2):
        self.policy = policy
        self.mode = mode
        self.width = width

    def set_seed(self, seed):
        self.policy.set_seed(seed)

    def check_observation(self, text):
        self.policy.check_observation(text)

    def complete(self, messages, **kwargs):
        if not kwargs.get("stop_at_tool", True):
            return self.policy.complete(messages, **kwargs)
        turns = [m for m in messages if m["role"] == "assistant"]
        used = sum(len(parse_batch(m["content"])) for m in turns)
        remaining = 4 - used
        if self.mode == "width-sweep":
            if not turns:
                widths, finish = (self.width,), False
            else:
                widths = tuple(range(1, self.width + 1)) if len(turns) == 1 else ()
                finish = True
        elif self.mode == "fixed-serial":
            widths = (1,) if remaining > 0 else ()
            finish = not widths
        elif self.mode == "fixed-parallel":
            widths = (2,) if remaining > 0 else ()
            finish = not widths
        else:
            # Seed the first pair, then let fresh observations govern further retrieval.
            widths = ((2,) if not turns else tuple(range(1, min(2, remaining) + 1))) if remaining > 0 else ()
            finish = bool(turns)
        messages = [dict(message) for message in messages]
        if not widths:
            instruction = "Finish now using the collected evidence, in one finish(answer) tool call."
        else:
            count = " or ".join(str(width) for width in widths)
            instruction = (
                f"Use exactly {count} search tool calls this turn."
                if not finish else
                f"Finish if the evidence answers the question. Otherwise use {count} search tool calls."
            )
            instruction += (
                " For each search query use entities and details from the actual question or new observations."
                " Look for missing evidence. Use complementary queries and avoid repeating past requests."
            )
        messages[-1]["content"] += (
            "\nFor this turn output only the <tool_calls> block using the original tool JSON entries. " + instruction
        )
        kwargs["tool_stop"] = BATCH_STOP
        return self.policy.complete(messages, guided_grammar=batch_grammar(widths, allow_finish=finish), **kwargs)


ARMS = ("fixed-parallel", "fixed-serial", "adaptive-parallel", "memt-serial")
WIDTHS = (2, 4, 8)


def compare_widths(output):
    results, rows = {}, {}
    query_keys = ("sample_id", "qa_id", "question", "gold", "category")
    for width in WIDTHS:
        arm = output / f"width-{width}"
        rows[width] = read_jsonl(arm / "answers.jsonl")
        values = rows[width]
        if not values:
            raise ValueError(f"No answers for width {width}")
        first_widths = Counter(row["traces"][0]["search_calls"] for row in values)
        results[width] = {
            **read_json(arm / "answers.metrics.json"), **round_diagnostics(values),
            "first_round_width_counts": dict(first_widths),
            "questions_using_full_first_round": first_widths[width],
            "mean_unique_memory_entries": sum(len(row["traces"][-1]["retrieved"]) for row in values) / len(values),
        }
    query_contract = [tuple(row[key] for key in query_keys) for row in rows[2]]
    if any([tuple(row[key] for key in query_keys) for row in rows[width]] != query_contract
           for width in WIDTHS[1:]):
        raise ValueError("Width sweep question contracts differ")

    contrasts = {}
    for small, large in zip(WIDTHS, WIDTHS[1:]):
        differences = [after["f1"] - before["f1"] for before, after in zip(rows[small], rows[large])
                       if before["f1"] is not None and after["f1"] is not None]
        contrasts[f"{large}_vs_{small}"] = {
            "f1_delta": results[large]["f1"] - results[small]["f1"],
            "mean_calls_delta": results[large]["mean_search_calls"] - results[small]["mean_search_calls"],
            "better": sum(delta > 0 for delta in differences),
            "worse": sum(delta < 0 for delta in differences),
            "tied": sum(delta == 0 for delta in differences),
        }
    paired_questions = [
        {"qa_id": row["qa_id"], "category": row["category"],
         **{f"width-{width}": {key: rows[width][idx][key] for key in
                             ("f1", "search_rounds", "search_calls", "latency_seconds")}
            for width in WIDTHS}}
        for idx, row in enumerate(rows[2])
    ]
    return {"widths": results, "contrasts": contrasts, "paired_questions": paired_questions}


def round_diagnostics(rows):
    handoffs = 0
    repeated_calls = 0
    by_parallel_rounds = Counter()
    for row in rows:
        seen = set()
        parallel = 0
        for node in row["traces"]:
            if not node["search_calls"]:
                continue
            actions = {json.dumps(call, sort_keys=True) for call in node["calls"]}
            repeated_calls += len(node["calls"]) - len(actions - seen)
            if node["search_calls"] > 1:
                if parallel and actions - seen:
                    handoffs += 1
                parallel += 1
            seen.update(actions)
        by_parallel_rounds[parallel] += 1
    return {
        "questions_by_parallel_rounds": dict(by_parallel_rounds),
        "questions_with_at_least_two_parallel_rounds": sum(n for k, n in by_parallel_rounds.items() if k >= 2),
        "later_parallel_rounds_with_a_new_call": handoffs,
        "repeated_search_calls": repeated_calls,
    }


def compare_arms(output):
    metrics, rows = {}, {}
    for name in ARMS:
        arm = output / name
        rows[name] = read_jsonl(arm / "answers.jsonl")
        metrics[name] = {**read_json(arm / "answers.metrics.json"), **round_diagnostics(rows[name])}
    baseline = rows["memt-serial"]
    fields = ("sample_id", "qa_id", "question", "gold", "category")
    identities = lambda values: [tuple(row[k] for k in fields) for row in values]
    golds = identities(baseline)
    if any(identities(values) != golds for values in rows.values()):
        raise ValueError("Question contract changed across experiment arms")

    contrasts = {}
    for left, right in (("fixed-serial", "fixed-parallel"), ("memt-serial", "adaptive-parallel")):
        before, after = metrics[left], metrics[right]
        contrasts[f"{right}_vs_{left}"] = {
            "f1_delta": after["f1"] - before["f1"],
            "p50_change_fraction": after["latency_p50_seconds"] / before["latency_p50_seconds"] - 1,
            "p95_change_fraction": after["latency_p95_seconds"] / before["latency_p95_seconds"] - 1,
            "mean_decisions_delta": after["mean_decisions"] - before["mean_decisions"],
        }
    multistep = [i for i, row in enumerate(baseline)
                 if row["search_rounds"] >= 2 and row["f1"] is not None]
    subset = {name: sum(values[i]["f1"] for i in multistep) / len(multistep)
              for name, values in rows.items()} if multistep else {}
    return {
        "arms": metrics, "contrasts": contrasts,
        "original_memt_multistep_subset": {"count": len(multistep), "f1": subset},
        "paired_questions": [
            {"qa_id": row["qa_id"], "category": row["category"],
             **{name: {k: rows[name][idx][k] for k in
                       ("f1", "search_rounds", "parallel_rounds", "search_calls", "latency_seconds")}
                for name in ARMS}}
            for idx, row in enumerate(baseline)
        ],
    }


# Pre-repair source digests from the sealed full LoCoMo experiment.
_HNSW_REPAIR_PREVIOUS = {
    "memory/journal.py": "fb50d5d8ebe06d1267ed5c7a26d6336e5af46acfc1af28c72eff55cfc8690bf4",
    "memory/vector_db.py": "1dbce13af2b4b2507bae94f4054dc522ca45c682a4e594f6e6f3c1edd2b02a11",
    "evaluation/guided_parallel.py": "798816adf620026dda777b8ee37a745b173ea3b0fb23da452d936b71abcb9040",
    "evaluation/locomo_full.py": "258b4ca7bbe4657c9339536abc698e4f6dd09c39184c0d235339f3a9b7af0525",
}


def validate_hnsw_recovery(original, repaired):
    original_meta = {k: v for k, v in original.items() if k != "code_sha256"}
    repaired_meta = {k: v for k, v in repaired.items() if k != "code_sha256"}
    old, new = original["code_sha256"], repaired["code_sha256"]
    changed = {p: old[p] for p in old.keys() & new.keys() if old[p] != new[p]}
    if original_meta != repaired_meta or old.keys() != new.keys() or changed != _HNSW_REPAIR_PREVIOUS:
        raise ValueError("HNSW recovery requires the original LoCoMo contract and the exact repair source delta")


def save_contract(output, contract, resume, recover_hnsw):
    manifest_path = output / "manifest.json"
    if recover_hnsw:
        validate_hnsw_recovery(read_json(manifest_path), contract)
        recovery = output / "hnsw-recovery.json"
        record = {"previous_manifest_sha256": sha256(manifest_path),
                  "database": "db-hnsw-recovery", "sync_threshold": 1, "manifest": contract}
        if recovery.exists() and read_json(recovery) != record:
            raise ValueError("HNSW recovery contract changed")
        write_json(recovery, record)
        return
    if resume and manifest_path.exists() and read_json(manifest_path) != contract:
        raise ValueError("Experiment contract changed")
    write_json(manifest_path, contract)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/memt4b.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--embedding", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recover-hnsw", action="store_true",
                        help="Resume the sealed full LoCoMo run with freshly persisted snapshot indexes")
    parser.add_argument("--width-sweep", action="store_true",
                        help="Compare first-batch widths 2, 4 and 8 with an optional follow-up round")
    parser.add_argument("--width-order", type=int, nargs=3, metavar="WIDTH",
                        help="Permutation of 2 4 8 for independent width-sweep repeats")
    args = parser.parse_args()
    if args.width_order and (not args.width_sweep or tuple(sorted(args.width_order)) != WIDTHS):
        parser.error("--width-order requires --width-sweep and a permutation of 2 4 8")
    if args.recover_hnsw and not (args.resume and args.width_sweep and args.width_order):
        parser.error("--recover-hnsw requires --resume, --width-sweep and --width-order")
    width_order = args.width_order or WIDTHS

    from experiments.workflow import MemoryRun, seed_all
    from models.vllm import VLLMPolicy

    cfg = load_config(args.config)
    cfg["model"]["checkpoint"] = str(Path(args.model).resolve())
    cfg["memory"]["embedding_model"] = str(Path(args.embedding).resolve())
    cfg["retrieval"].update(expansion_rounds=1, evidence_alpha=1)
    cfg["distributed"].update(gpus=2, tensor_parallel=2, inference_tensor_parallel=2,
                              gpu_memory_utilization=0.55)
    cfg["runtime"] = {"resumable_evaluation": True}
    cfg["runtime"]["experiment_prefix_cache"] = False
    cfg["runtime"]["experiment_eager"] = True
    if args.width_sweep:
        cfg["evaluation"]["max_steps"] = 3
        cfg["runtime"]["width_sweep"] = list(WIDTHS)
        if args.width_order:
            cfg["runtime"]["width_order"] = args.width_order
    output = checked_output(args.output, resume=args.resume)
    seed_all(cfg["seed"])
    database = "db-hnsw-recovery" if args.recover_hnsw else "db"
    memory = MemoryRun(args.memory, cfg, db_path=output / database)
    policy = VLLMPolicy(cfg["model"]["checkpoint"], cfg, enforce_eager=True, enable_prefix_caching=False)
    contract = manifest(cfg, [args.memory / "index.json", args.memory / "questions.jsonl"])
    save_contract(output, contract, args.resume, args.recover_hnsw)

    # Warm up each grammar before timing any question.
    warmup = [((), True), ((1,), False), ((2,), False), ((1, 2), True), ((1,), True)]
    if args.width_sweep:
        warmup = [((), True)] + [(widths, finish) for width in WIDTHS
                                for widths, finish in (((width,), False), (tuple(range(1, width + 1)), True))]
    for widths, allow_finish in warmup:
        if widths:
            prompt = "Search the stored memories for the dates of the two earlier project meetings."
        else:
            prompt = "Finish with the answer Paris."
        policy.complete([{"role": "user", "content": prompt}], temperature=0, max_new_tokens=512 if args.width_sweep else 96,
                        tool_stop=BATCH_STOP, guided_grammar=batch_grammar(widths, allow_finish=allow_finish))
    arms = [(f"width-{width}", width) for width in width_order] if args.width_sweep else [
        (name, 1 if name.endswith("serial") else 2) for name in ARMS
    ]
    for name, width in arms:
        arm_cfg = copy.deepcopy(cfg)
        arm_cfg["evaluation"]["max_parallel_calls"] = width
        arm = output / name
        arm.mkdir(exist_ok=args.resume)
        if args.width_sweep:
            arm_cfg["evaluation"]["max_search_calls"] = 2 * width
            current = GuidedPolicy(policy, "width-sweep", width=width)
        else:
            current = policy if name == "memt-serial" else GuidedPolicy(policy, name)
        write_json(output / "state.json", {"status": "running", "arm": name})
        result = evaluate(arm_cfg, memory, current, arm / "answers.jsonl")
        print(name, json.dumps(result), flush=True)
    report = compare_widths(output) if args.width_sweep else compare_arms(output)
    write_json(output / "report.json", report)
    write_json(output / "state.json", {"status": "complete"})


if __name__ == "__main__":
    main()
