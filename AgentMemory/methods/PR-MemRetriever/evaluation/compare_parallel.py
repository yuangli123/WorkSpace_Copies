"""Compare evaluations with the same questions, snapshot, model and budgets."""
import argparse
import copy
import json
from pathlib import Path

from evaluation.retrieval import metric_summary
from experiments.io import read_json, read_jsonl, write_json


def _read_run(root):
    paths = [path for path in (root / "answers.jsonl", root / "answers.jsonl.gz") if path.is_file()]
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one answers file under {root}")
    rows = read_jsonl(paths[0])
    by_id = {(row["sample_id"], row["qa_id"]): row for row in rows}
    if not rows or len(rows) != len(by_id):
        raise ValueError(f"No rows or duplicate question identities under {root}")
    return read_json(root / "manifest.json"), by_id


def _comparable(manifest):
    config = copy.deepcopy(manifest["config"])
    evaluation = config["evaluation"]
    evaluation.pop("max_parallel_calls", None)
    evaluation.pop("serial_tool_execution", None)
    return config, manifest["inputs"], manifest["code_sha256"]


def compare(reference, candidate):
    ref_manifest, left = _read_run(Path(reference))
    cand_manifest, right = _read_run(Path(candidate))
    if _comparable(ref_manifest) != _comparable(cand_manifest):
        raise ValueError("Evaluation contract differs beyond the parallel mode")
    if left.keys() != right.keys():
        raise ValueError("Question identities differ between evaluations")
    pairs = []
    fields = ("f1", "bleu1", "latency_seconds", "tool_seconds", "search_calls", "steps")
    for key, old in left.items():
        new = right[key]
        if any(old[name] != new[name] for name in ("question", "gold", "category")):
            raise ValueError(f"Question contents differ: {key}")
        pairs.append({"sample_id": key[0], "qa_id": key[1], "category": old["category"],
                      **{name: new[name] - old[name] for name in fields
                         if new[name] is not None and old[name] is not None}})
    baseline = metric_summary(list(left.values()))
    parallel = metric_summary(list(right.values()))
    categories = {}
    for row in pairs:
        if "f1" in row:
            categories.setdefault(str(row["category"]), []).append(row["f1"])
    return {"reference": baseline, "candidate": parallel,
            "delta": {name: parallel[name] - value for name, value in baseline.items()
                      if value is not None and parallel[name] is not None},
            "category_f1_delta": {k: sum(v) / len(v) for k, v in categories.items()},
            "per_question_delta": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = compare(args.reference, args.candidate)
    if args.output:
        write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "per_question_delta"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
