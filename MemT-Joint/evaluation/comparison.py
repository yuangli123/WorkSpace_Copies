import random

from experiments.io import read_jsonl


def compare(paths, bootstrap=2000, seed=42):
    methods = {}
    for name, path in paths.items():
        rows = read_jsonl(path)
        indexed = {r['qa_id']: r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError('Duplicate QA IDs in ' + name)
        methods[name] = indexed
    names = list(methods)
    protocol_ids = {row.get('protocol_id') for rows in methods.values() for row in rows.values()}
    if len(protocol_ids) != 1:
        raise ValueError('Inference protocols differ; compare only matched tool/Answer/budget/failure/backend conditions')
    first = methods[names[0]]
    for name in names[1:]:
        if set(methods[name]) != set(first):
            raise ValueError('Mismatched question coverage')
        for key, row in methods[name].items():
            if any(row[k] != first[key][k] for k in ('sample_id', 'question', 'answer')):
                raise ValueError('Mismatched question or gold content')
    result = {name: {'count': len(rows), 'f1': sum(r['f1'] for r in rows.values()) / len(rows)} for name, rows in methods.items()}
    clusters = sorted({r['sample_id'] for r in first.values()})
    rng = random.Random(seed)
    for name in names[1:]:
        differences = []
        for _ in range(bootstrap):
            selected = rng.choices(clusters, k=len(clusters))
            ids = [key for sid in selected for key, row in first.items() if row['sample_id'] == sid]
            differences.append(sum(methods[name][key]['f1'] - first[key]['f1'] for key in ids) / len(ids))
        differences.sort()
        result[name]['paired_cluster_bootstrap_delta_ci95'] = [differences[int(.025 * bootstrap)], differences[min(bootstrap - 1, int(.975 * bootstrap))]]
    return result
