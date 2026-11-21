"""MemT-Joint entry point. Imports heavy training dependencies only on demand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--benchmark', choices=['locomo', 'longmemeval'], default='locomo')
    for name in ('preflight', 'smoke', 'train', 'evaluate'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--config', default='configs/joint.json')
        cmd.add_argument('--data', required=True)
        if name != 'preflight':
            cmd.add_argument('--output', required=True)
        if name in ('train', 'smoke'):
            cmd.add_argument('--resume', action='store_true')
            cmd.add_argument('--probes')
            cmd.add_argument('--valid-data')
        if name in ('train', 'evaluate'):
            cmd.add_argument('--variant', choices=['M0', 'M1', 'M2', 'M3', 'M4', 'M5'])
        if name == 'evaluate':
            cmd.add_argument('--model', help='Joint policy HF export used for both retrieval and finish(answer)')
            cmd.add_argument('--mode', choices=['source_backed', 'memory_only'])
            cmd.add_argument('--construction-model')
            cmd.add_argument('--retrieval-model')
    export = sub.add_parser('export')
    export.add_argument('--run', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--selection', choices=['best', 'latest'], default='best')
    probe = sub.add_parser('verify-restore')
    probe.add_argument('--run', required=True)
    probe.add_argument('--output', required=True)
    comparison = sub.add_parser('compare')
    comparison.add_argument('--answers', nargs='+', required=True, help='name=answers.jsonl')
    comparison.add_argument('--output', required=True)
    collection = sub.add_parser('collect-sft')
    collection.add_argument('--config', default='configs/two-stage.json')
    collection.add_argument('--data', required=True)
    collection.add_argument('--retrieval-model', required=True)
    collection.add_argument('--output', required=True)
    sft = sub.add_parser('sft')
    sft.add_argument('--config', default='configs/two-stage.json')
    sft.add_argument('--selected', required=True)
    sft.add_argument('--output', required=True)
    sft.add_argument('--resume')
    judge = sub.add_parser('judge')
    judge.add_argument('--config', default='configs/joint.json')
    judge.add_argument('--answers', required=True)
    judge.add_argument('--output', required=True)
    judge.add_argument('--protocol', choices=['legacy-topic-v1', 'memory-r2-generous-v1', 'longmemeval-official-prompt'], required=True)
    judge.add_argument('--model')
    judge.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    from experiments.io import write_json, read_json, checked_output
    if args.command == 'verify-restore':
        from training.restore_probe import verify
        print(json.dumps(verify(args.run, args.output)))
        return
    if args.command == 'prepare':
        from memory.data import prepare_locomo, prepare_ood
        if args.benchmark == 'locomo':
            prepare_locomo(args.source, args.output)
        else:
            prepare_ood(args.source, args.output, 'longmemeval')
        return
    if args.command == 'compare':
        from evaluation.comparison import compare
        write_json(args.output, compare(dict(item.split('=', 1) for item in args.answers)))
        return
    if args.command == 'export':
        from experiments.joint import restore, checkpoint_path
        run = Path(args.run)
        cfg = read_json(run / 'manifest.json')['config']
        if cfg['backend'] == 'fixture':
            raise ValueError('Fixture smoke has no trained model')
        if cfg['backend'] == 'verl':
            from training.verl_backend import export_checkpoint
            checkpoint = checkpoint_path(run, args.selection)
            from experiments.io import sha256
            for name, value in read_json(checkpoint / 'RECEIPT.json').items():
                if sha256(checkpoint / name) != value:
                    raise ValueError('Modified checkpoint')
            export_checkpoint(checkpoint, args.output)
        else:
            from models.hf import HFPolicy
            from training.native import NativeUpdater
            if cfg['experiment'].get('variant') == 'M5':
                from training.separate import create
                policy, updater = create(cfg, training=True)
            else:
                policy = HFPolicy(cfg['models']['base'], cfg)
                updater = NativeUpdater(policy, cfg)
            restore(updater, run, args.selection)
            target = checked_output(args.output)
            updater.export(target)
            write_json(target / 'COMPLETE.json', {'run': str(run.resolve())})
        return
    from experiments.joint_config import load, validate, variant
    cfg = load(args.config)
    if args.command == 'judge':
        from evaluation.joint_judge import score
        print(json.dumps(score(cfg, args.answers, args.output, args.protocol, args.model, args.resume)))
        return
    if args.command == 'collect-sft':
        from experiments.two_stage import collect_sft
        print(json.dumps(collect_sft(cfg, args.data, args.retrieval_model, args.output)))
        return
    if args.command == 'sft':
        from experiments.io import read_jsonl
        from training.sft import train_sft
        train_sft(cfg, read_jsonl(args.selected), cfg['models']['base'], args.output, args.resume)
        return
    if getattr(args, 'variant', None):
        cfg = variant(cfg, args.variant)
    validate(cfg, training=args.command == 'train')
    from experiments.joint import identity, corpus, run, evaluate
    if args.command == 'preflight':
        episodes, questions = corpus(args.data)
        print(json.dumps({'episodes': len(episodes), 'questions': len(questions), 'identity': identity(cfg, args.data)}, ensure_ascii=False))
    elif args.command in ('train', 'smoke'):
        run(cfg, args.data, args.output, probes=args.probes, resume=args.resume, smoke=args.command == 'smoke', valid_data=args.valid_data)
    else:
        print(json.dumps(evaluate(cfg, args.data, args.output, args.model, args.mode, args.construction_model, args.retrieval_model), ensure_ascii=False))


if __name__ == '__main__':
    main()
