"""Render small scalar reports without loading rollout forests or model weights."""
import argparse
import copy
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path


def summarize(rows, total):
    steps = [row['step'] for row in rows]
    assert steps == list(range(1, len(rows) + 1)), 'Missing or duplicate step metrics'
    assert all(row['global_query_batch'] == 32 for row in rows)
    assert all(math.isfinite(row['wall_seconds']) and row['wall_seconds'] >= 0 for row in rows)
    recent = rows[-10:]
    compute = statistics.median(row['wall_seconds'] for row in recent) if recent else None
    intervals = [b['resources']['time_unix'] - a['resources']['time_unix']
                 for a, b in zip(recent, recent[1:])]
    assert all(value > 0 for value in intervals), 'Non-increasing step completion timestamps'
    wall = statistics.mean(intervals) if intervals else None
    return dict(completed_steps=len(rows), target_steps=total,
                recent_compute_seconds_median=compute,
                recent_wall_seconds_mean=wall,
                estimated_remaining_update_hours=(total - len(rows)) * compute / 3600 if compute and len(rows) >= 5 else None,
                estimated_remaining_observed_cadence_hours=(total - len(rows)) * wall / 3600 if wall and len(rows) >= 5 else None,
                eta_scope='At least five real updates are required for an ETA. '
                          'Update ETA excludes saves, validation, initialization, export and final test. '
                          'Observed cadence includes only overhead already encountered; first validation is step 20.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        rows = [dict(step=i, global_query_batch=32, wall_seconds=10, resources={'time_unix': i * 15})
                for i in range(1, 6)]
        summary = summarize(rows, 200)
        assert summary['completed_steps'] == 5 and summary['recent_wall_seconds_mean'] == 15
        assert summary['estimated_remaining_update_hours'] == 1950 / 3600
        assert summarize([], 200)['estimated_remaining_update_hours'] is None
        assert summarize(rows[:1], 200)['estimated_remaining_update_hours'] is None
        try:
            summarize(rows[1:], 200)
        except AssertionError:
            pass
        else:
            raise AssertionError('Missing step was accepted')
        print('report self-test passed')
        return
    assert args.run and args.config
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from experiments.io import configure_storage, get_budget, load_config, read_json, write_json
    cfg = load_config(args.config)
    run = args.run.resolve()
    assert run.is_relative_to(root / 'artifacts')
    report = run / 'report'
    observer = copy.deepcopy(cfg)
    observer['storage']['ledger'] = str(run / 'storage-report.json')
    configure_storage(observer)
    get_budget().check(25_000_000, 'scalar report and curves')
    report.mkdir(parents=True, exist_ok=True)
    rows = [read_json(file) for file in sorted((run / 'retrieval/metrics').glob('step-*.json'))]
    summary = summarize(rows, cfg['retrieval']['steps'])
    summary['resource_peaks_from_steps'] = {
        key: max(row.get('resource_peaks', {}).get(key, 0) for row in rows)
        for key in set().union(*(row.get('resource_peaks', {}).keys() for row in rows))}
    summary['completed_monitor_peaks'] = {}
    for folder in (run / 'runner/telemetry', run / 'retrieval'):
        peaks, started = folder / 'resources/peaks.json', folder / 'RUNNING.json'
        if peaks.exists() and started.exists() and peaks.stat().st_mtime >= started.stat().st_mtime:
            summary['completed_monitor_peaks'][str(folder.relative_to(run))] = read_json(peaks)
    valid = [dict(step=int(file.name.split('-')[1].split('.')[0]), **read_json(file))
             for file in sorted((run / 'retrieval/validation').glob('step-*.metrics.json'))]
    summary['validation'] = valid
    summary['storage'] = {file.name: read_json(file) for file in run.glob('storage-*.json')
                          if file.name != 'storage-report.json'}
    scalar = [{key: value for key, value in row.items() if isinstance(value, (int, float))} for row in rows]
    if scalar:
        columns = ['step'] + sorted(set().union(*(row.keys() for row in scalar)) - {'step'})
        with (report / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(scalar)
        os.environ.setdefault('MPLCONFIGDIR', str(report / 'matplotlib-cache'))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
        steps = [row['step'] for row in rows]
        for axis, key in zip(axes.flat, ('actor/loss', 'actor/update_lr', 'mean_reward',
                                         'leaf_performance', 'wall_seconds', 'storage_observed_bytes')):
            values = [row.get(key, float('nan')) for row in rows]
            if key == 'storage_observed_bytes':
                values = [value / 1e9 for value in values]
                axis.axhline(100, color='red', linestyle='--', label='100 GB budget')
                axis.set_ylabel('GB (decimal)')
            axis.plot(steps, values, marker='.', label=key)
            if key == 'leaf_performance' and valid:
                axis.plot([row['step'] for row in valid], [row['f1'] for row in valid], 'o-', label='validation F1')
            axis.set_title(key)
            axis.set_xlabel('Optimizer update')
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        figure.savefig(report / 'training.png', dpi=160)
        figure.savefig(report / 'training.pdf')
        plt.close(figure)
    for name in ('BEST.json', 'TRAINING_COMPLETE.json', 'RESTORE_VERIFIED.json'):
        file = run / 'retrieval' / name
        if file.exists():
            summary[name] = read_json(file)
    write_json(report / 'summary.json', summary)
    text = ('# Mem-T batch32 experiment\n\n'
            f"Completed optimizer updates: {len(rows)}/{cfg['retrieval']['steps']}. Global query batch: 32.\n\n"
            f"Recent update median seconds: {summary['recent_compute_seconds_median']}. "
            f"Recent completion cadence mean seconds: {summary['recent_wall_seconds_mean']}.\n\n"
            f"Remaining update-only hours: {summary['estimated_remaining_update_hours']}. "
            f"Remaining observed-cadence hours: {summary['estimated_remaining_observed_cadence_hours']}.\n\n"
            + summary['eta_scope'] + '\n\n'
            'Research settings: M=1, alpha=1, BF16 actor and AdamW. Frozen-base memory construction; '
            'this run trains retrieval only. Test data do not select the best model.\n\n'
            'Disk accounting uses decimal bytes and includes project files, base models and caches; '
            'the existing Conda environment is separately excluded. Admission reserves are estimates, '
            'not a filesystem quota. Resource peaks use sampled current memory, not inherited cgroup maxima.\n')
    (report / 'REPORT.md').write_text(text, encoding='utf-8')
    print(json.dumps({key: value for key, value in summary.items() if key != 'storage'}, indent=2))


if __name__ == '__main__':
    main()
