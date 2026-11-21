from __future__ import annotations

import math
from collections import defaultdict


def hindsight_scores(operations, records, questions, config):
    """Equations 16-17. Average leaves within a query, then queries per episode."""
    question_map = {q["qa_id"]: q for q in questions}
    by_episode = defaultdict(list)
    for index, operation in enumerate(operations):
        by_episode[operation['sample_id']].append(index)
    per_query = [[] for _ in operations]
    evidence_hits = [0] * len(operations)
    trace_hits = [0] * len(operations)
    # The final post-training pass uses one policy version and one forest per QA.
    seen = set()
    for record in records:
        if record["qa_id"] in seen:
            raise ValueError("Duplicate QA forests: use a fixed final-policy collection, not mixed RL iterations")
        seen.add(record["qa_id"])
        if record["qa_id"] not in question_map:
            raise ValueError("Unknown QA in hindsight records")
        if record["sample_id"] != question_map[record["qa_id"]]["sample_id"]:
            raise ValueError("Hindsight forest belongs to a different episode")
        qa = question_map[record['qa_id']]
        leaves = [node for tree in record['trees'] for node in tree[1:] if not node['children']]
        retrieved = [{op for m in leaf['retrieved'] for op in m.get('operation_ids', [])} for leaf in leaves]
        for index in by_episode[record['sample_id']]:
            operation = operations[index]
            evidence = bool(set(operation["source_turn_ids"]) & set(qa.get("evidence", [])))
            values = []
            for leaf, retrieved_ops in zip(leaves, retrieved):
                trace = operation["op_id"] in retrieved_ops
                coefficient = (float(evidence) if config["construction"]["evidence_gate"] else 0.0)
                coefficient += config["construction"]["trace_lambda"] * float(trace) if config["construction"]["trace_gate"] else 0.0
                values.append(leaf["advantage"] * coefficient)
                evidence_hits[index] += int(evidence)
                trace_hits[index] += int(trace)
            if values:
                per_query[index].append(sum(values) / len(values))
    return [dict(operation, score=sum(per_query[i]) / len(per_query[i]) if per_query[i] else None,
                 evidence_hits=evidence_hits[i], trace_hits=trace_hits[i], scored_queries=len(per_query[i]))
            for i, operation in enumerate(operations)]


def select_operations(scored, fraction=0.5, maximum=10000):
    if not 0 < fraction <= 1:
        raise ValueError("Selection fraction must be in (0,1]")
    groups = defaultdict(list)
    invalid_completions = {operation.get('completion_id') for operation in scored
                           if not operation['valid'] or operation.get('completion_valid') is False}
    invalid_completions.discard(None)
    for operation in scored:
        score = operation.get("score")
        if (operation["valid"] and operation.get('completion_id') not in invalid_completions
                and score is not None and math.isfinite(score)):
            groups[operation["phase"] + ":" + operation["category"]].append(operation)
    selected, audit = [], {}
    for category, values in sorted(groups.items()):
        values.sort(key=lambda op: (-op["score"], op["op_id"]))
        # Small groups retain ceil(n/2); ties are deterministic, never all retained.
        keep = max(1, math.ceil(len(values) * fraction))
        selected.extend(values[:keep])
        audit[category] = {"eligible": len(values), "selected": keep,
                           "cutoff": values[keep - 1]["score"],
                           "distinct_scores": len({v["score"] for v in values})}
    # A global cap is a reproducibility choice; don't silently alter category fractions.
    if len(selected) > maximum:
        raise ValueError(f"Selected {len(selected)} operations exceed cap {maximum}; increase explicit cap or collect a fixed subset first")
    return selected, audit
