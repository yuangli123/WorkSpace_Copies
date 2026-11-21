"""Validate aligned answer/score files before writing a comparison table."""

import argparse
import hashlib
import json
import os
from collections import defaultdict

from evaluation.protocol import atomic_json, load_selected, question_text
from evaluation.measurement import summarize


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def validate_rows(path, expected, scored=False):
    rows = read_json(path)
    if set(rows) != {str(i) for i in range(len(expected))}:
        raise ValueError(f"Missing or extra conversations: {path}")
    for index, questions in enumerate(expected):
        actual = rows[str(index)]
        if len(actual) != len(questions):
            raise ValueError(f"Question count mismatch: {path}, conversation {index}")
        for position, question in enumerate(questions):
            item = actual[position]
            for key, value in (("question", question_text(question)),
                               ("answer", str(question.answer) if scored else question.answer),
                               ("category", question.category)):
                if item.get(key) != value:
                    raise ValueError(f"{key} mismatch: {path}, conversation {index}, question {position}")
            if not isinstance(item.get("response"), str):
                raise ValueError(f"response missing or not text: {path}, conversation {index}, question {position}")
            if not scored and item.get("answerable") != question.answerable:
                raise ValueError(f"answerable mismatch: {path}, conversation {index}, question {position}")
    return rows


def compare(bench, data_path, runs, output, conv_limit=None, sample_size=None, overwrite=False,
            with_metrics=False):
    if os.path.exists(output) and not overwrite:
        raise FileExistsError(output)
    _, questions, ids = load_selected(data_path, bench, conv_limit, sample_size)
    digest = hashlib.sha256()
    with open(data_path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    dataset_sha = digest.hexdigest()
    table = []
    judge_models = set()
    generation_models = set()
    embedding_models = set()
    efficiency = []
    for method, answer_path, score_path in runs:
        answer_manifest = read_json(answer_path + ".run.json")
        score_manifest = read_json(score_path + ".run.json")
        if answer_manifest.get("method") != method or answer_manifest.get("bench") != bench:
            raise ValueError(f"Run identity mismatch: {answer_path}")
        if answer_manifest.get("dataset_sha256") != dataset_sha or score_manifest.get("dataset_sha256") != dataset_sha:
            raise ValueError(f"Dataset hash mismatch: {method}")
        if answer_manifest.get("conv_limit") != conv_limit or answer_manifest.get("sample_size") != sample_size:
            raise ValueError(f"Question selection mismatch: {method}")
        if answer_manifest.get("question_ids", ids) != ids:
            raise ValueError(f"Question IDs mismatch: {method}")
        if score_manifest.get("bench") != bench or os.path.realpath(answer_path) != score_manifest.get("answer_file"):
            raise ValueError(f"Score source mismatch: {method}")
        with open(answer_path, "rb") as handle:
            answer_sha = hashlib.sha256(handle.read()).hexdigest()
        if score_manifest.get("answer_sha256") != answer_sha:
            raise ValueError(f"Answer changed after scoring: {method}")
        judge_models.add(score_manifest.get("judge_model"))
        generation_models.add((answer_manifest.get("model"), answer_manifest.get("llm_base_url")))
        if method != "fullcontext":
            embedding_models.add((answer_manifest.get("embedding_model"), answer_manifest.get("embedding_base_url")))
        answers = validate_rows(answer_path, questions)
        scores = validate_rows(score_path, questions, scored=True)
        if with_metrics:
            if answer_manifest.get("measure") is not True or answer_manifest.get("measurement_schema") != 1:
                raise ValueError(f"Run was not measured: {method}")
            expected = {str(i): len(items) for i, items in enumerate(questions)}
            measured = summarize(answer_path + ".metrics.json", expected)
            context_path = answer_path + ".context.json"
            if os.path.exists(context_path):
                context = read_json(context_path)
                if set(context) != set(expected) or any(len(context[key]) != count
                                                       for key, count in expected.items()):
                    raise ValueError(f"Context statistics mismatch: {method}")
                measured["truncated_questions"] = sum(
                    bool(item.get("removed_chunks", 0) or item.get("removed_tokens_estimate", 0))
                    for rows in context.values() for item in rows
                )
            else:
                measured["truncated_questions"] = None
            efficiency.append({"method": method, **measured})
        for index in range(len(questions)):
            for position, answer in enumerate(answers[str(index)]):
                if answer["response"] != scores[str(index)][position]["response"]:
                    raise ValueError(f"Scored response differs: {method}, {index}, {position}")
        categories = defaultdict(list)
        for row in scores.values():
            for item in row:
                for metric in ("llm_score", "f1_score", "bleu_score"):
                    if not isinstance(item.get(metric), (int, float)):
                        raise ValueError(f"Missing or nonnumeric {metric}: {score_path}")
                categories["overall"].append(item)
                categories[str(item["category"])].append(item)
        for category, items in categories.items():
            table.append({
                "method": method, "category": category, "count": len(items),
                **{metric: sum(item[metric] for item in items) / len(items)
                   for metric in ("llm_score", "f1_score", "bleu_score")},
            })
    if len(judge_models) != 1 or None in judge_models:
        raise ValueError("Judge models differ or are unrecorded")
    if len(generation_models) != 1 or len(embedding_models) > 1:
        raise ValueError("Generation or embedding model configuration differs")
    if any(not value for value in next(iter(generation_models))) or any(
        not value for model in embedding_models for value in model
    ):
        raise ValueError("Generation or embedding endpoint is unrecorded")
    result = {"bench": bench, "dataset_sha256": dataset_sha,
              "judge_model": judge_models.pop(),
              "generation_model": next(iter(generation_models))[0],
              "embedding_model": next(iter(embedding_models))[0] if embedding_models else None,
              "question_ids": ids, "scores": table}
    if with_metrics:
        result["efficiency"] = efficiency
    atomic_json(output, result)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, choices=["locomo", "longmemeval_s"])
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--run", action="append", required=True,
                        help="METHOD=ANSWERS.json=SCORES.json; repeat for each method")
    parser.add_argument("--output", required=True)
    parser.add_argument("--conv_limit", type=int)
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--metrics", action="store_true", help="Validate and include measured efficiency")
    args = parser.parse_args(argv)
    runs = []
    for value in args.run:
        parts = value.split("=", 2)
        if len(parts) != 3:
            parser.error("--run requires METHOD=ANSWERS.json=SCORES.json")
        runs.append(tuple(parts))
    if len({run[0] for run in runs}) != len(runs):
        parser.error("method names must be unique")
    compare(args.bench, args.data_path, runs, args.output, args.conv_limit,
            args.sample_size, args.overwrite, args.metrics)


if __name__ == "__main__":
    main()
