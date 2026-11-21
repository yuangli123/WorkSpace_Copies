"""Prepare all ten LoCoMo conversations and aggregate three width-sweep repeats."""
import argparse
import copy
import statistics
from collections import Counter
from pathlib import Path

from evaluation.retrieval import metric_summary
from experiments.io import checked_output, iter_jsonl, read_json, read_jsonl, sha256, write_json, write_jsonl

WIDTHS = (2, 4, 8)
SPLITS = ("train", "valid", "test")
EXPECTED = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
WIDTH_ORDERS = ((2, 4, 8), (4, 8, 2), (8, 2, 4))


def prepare(source, legacy, output):
    source_hash = sha256(source)
    conversations = read_json(source)
    index, scored, split_of, old_operations = {}, {}, {}, []
    for split in SPLITS:
        root = (legacy / f"memory-{split}").resolve()
        partition = read_json(root / "partition.json")
        if partition["source_sha256"] != source_hash or partition["split"] != split:
            raise ValueError(f"Snapshot source or split mismatch: {root}")
        expected_ids = {f"{sample_id}__r0" for sample_id in partition["sample_ids"]}
        entries = read_json(root / "index.json")
        if set(entries) != expected_ids or expected_ids & set(index):
            raise ValueError(f"Snapshot identity mismatch: {root}")
        read_json(root / "COMPLETE.json")
        for key, entry in entries.items():
            snapshot = (root / entry["snapshot"]).resolve()
            snapshot.relative_to(root)
            if sha256(snapshot) != entry["snapshot_sha256"]:
                raise ValueError(f"Snapshot checksum mismatch: {snapshot}")
            index[key] = {**entry, "snapshot": str(snapshot)}
            split_of[key] = split
        for row in iter_jsonl(root / "questions.jsonl"):
            key = row["qa_id"]
            if key in scored or row["sample_id"] not in expected_ids:
                raise ValueError(f"Duplicate question or wrong snapshot: {key}")
            scored[key] = row
        gz = root / "operations.jsonl.gz"
        old_operations.append(gz if gz.exists() else root / "operations.jsonl")

    questions = []
    for convo in conversations:
        sample_id = f'{convo["sample_id"]}__r0'
        for offset, row in enumerate(convo["qa"]):
            key = f'{convo["sample_id"]}:{offset}__r0'
            category = int(row["category"])
            if category == 5:
                question = {"sample_id": sample_id, "qa_id": key, "question": row["question"],
                            "answer": "No information available", "category": 5, "benchmark": "locomo",
                            "evidence": [], "evaluation_metric": "locomo_abstention"}
            else:
                question = scored.pop(key)
                if (question["sample_id"], question["question"], question["answer"], int(question["category"])) != (
                    sample_id, row["question"], row["answer"], category
                ):
                    raise ValueError(f"Raw and snapshot question disagree: {key}")
            questions.append({**question, "legacy_split": split_of[sample_id]})

    counts = Counter(int(row["category"]) for row in questions)
    if scored or len(index) != 10 or counts != EXPECTED:
        raise ValueError(f"Incomplete LoCoMo coverage: {counts}")
    if len({row["qa_id"] for row in questions}) != len(questions):
        raise ValueError("Duplicate prepared QA identity")
    output = checked_output(output)
    write_json(output / "index.json", index)
    write_jsonl(output / "questions.jsonl", questions)
    write_jsonl(output / "operations.jsonl.gz", (row for path in old_operations for row in iter_jsonl(path)))
    write_json(output / "partition.json", {"split": "full-eval", "source_sha256": source_hash,
                                           "sample_ids": sorted(index)})
    result = {"snapshot_count": len(index), "questions": len(questions), "by_category": dict(counts),
              "by_split": {split: dict(Counter(int(q["category"]) for q in questions if q["legacy_split"] == split))
                           for split in SPLITS}}
    write_json(output / "COMPLETE.json", result)
    print(result)


def job_contract(manifest):
    config = copy.deepcopy(manifest["config"])
    config["runtime"].pop("width_order")
    return config, manifest["inputs"], manifest["code_sha256"], manifest["packages"]


def summarize(run, memory, output):
    questions = read_jsonl(memory / "questions.jsonl")
    if Counter(int(q["category"]) for q in questions) != EXPECTED:
        raise ValueError("Unexpected LoCoMo corpus")
    reference = None
    runs = []
    for repeat, expected_order in enumerate(WIDTH_ORDERS, 1):
        root = run / f"repeat-{repeat}"
        state = read_json(root / "state.json")
        original_manifest = root / "manifest.json"
        manifest = read_json(original_manifest)
        recovery_path = root / "hnsw-recovery.json"
        recovery = None
        if recovery_path.exists():
            recovery = read_json(recovery_path)
            if recovery["previous_manifest_sha256"] != sha256(original_manifest):
                raise ValueError(f"Original manifest changed on repeat {repeat}")
            from evaluation.guided_parallel import validate_hnsw_recovery
            validate_hnsw_recovery(manifest, recovery["manifest"])
            manifest = recovery.pop("manifest")
        if state["status"] != "complete" or tuple(manifest["config"]["runtime"]["width_order"]) != expected_order:
            raise ValueError(f"Incomplete or reordered repeat: {repeat}")
        contract = job_contract(manifest)
        if reference is None:
            reference = contract
        elif contract != reference:
            raise ValueError(f"Experiment contract changed on repeat {repeat}")
        for path, digest in manifest["inputs"].items():
            if sha256(path) != digest:
                raise ValueError(f"Evaluation input changed: {path}")
        receipts = read_json(run / f"repeat-{repeat}-job.json")
        if receipts["repeat"] != repeat:
            raise ValueError(f"Job receipt mismatch: {repeat}")
        width_report = read_json(root / "report.json")
        by_width = {}
        for width in WIDTHS:
            rows = read_jsonl(root / f"width-{width}" / "answers.jsonl")
            if len(rows) != len(questions):
                raise ValueError(f"Incomplete width {width} on repeat {repeat}")
            for row, q in zip(rows, questions):
                keys = ("sample_id", "qa_id", "category", "question")
                if any(row[k] != q[k] for k in keys) or row["gold"] != q["answer"]:
                    raise ValueError(f"Question identity mismatch: {row['qa_id']}")
                if int(q["category"]) == 5:
                    if row["f1"] is not None or "abstention_correct" not in row:
                        raise ValueError(f"Adversarial result mixed with F1: {row['qa_id']}")
                elif row["f1"] is None or "abstention_correct" in row:
                    raise ValueError(f"F1 score missing: {row['qa_id']}")
            selection = {s: [r for r, q in zip(rows, questions) if q["legacy_split"] == s] for s in SPLITS}
            scopes = {name: metric_summary(value) for name, value in {"all": rows, **selection}.items()}
            scopes["all"]["questions_using_full_first_round"] = width_report["widths"][str(width)][
                "questions_using_full_first_round"]
            by_width[str(width)] = scopes
        entry = {"repeat": repeat, "job": receipts, "width_order": list(expected_order), "widths": by_width}
        if recovery:
            entry["hnsw_recovery"] = recovery
        runs.append(entry)

    if len({str(r["job"]["job_id"]) for r in runs}) != 3:
        raise ValueError("Repeat receipts do not identify three independent jobs")
    summary = {}
    for width in map(str, WIDTHS):
        widths = [r["widths"][width] for r in runs]
        summary[width] = {}
        for scope in ("all", *SPLITS):
            planned = ("f1", "bleu1", "abstention_accuracy", "mean_search_calls",
                       "latency_p50_seconds", "latency_p95_seconds", "finish_rate")
            metrics = {}
            for metric in planned:
                values = [r[scope][metric] for r in widths]
                metrics[metric] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                                   "repeats": values}
            summary[width][scope] = metrics
    write_json(output, {"questions": len(questions), "repeats": runs, "summary": summary})
    for width in map(str, WIDTHS):
        print(width, "all F1", summary[width]["all"]["f1"], "heldout F1", summary[width]["test"]["f1"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--legacy", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    summ = subs.add_parser("summarize")
    summ.add_argument("--run", type=Path, required=True)
    summ.add_argument("--memory", type=Path, required=True)
    summ.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.legacy, args.output)
    else:
        summarize(args.run, args.memory, args.output)


if __name__ == "__main__":
    main()
