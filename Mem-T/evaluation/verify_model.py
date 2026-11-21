"""Verify a real HF export, parameter changes, and reloaded construction tools."""
import argparse
from pathlib import Path

from experiments.io import load_config, write_json
from experiments.tiny_resume import tensor_files


def main():
    import torch
    from safetensors import safe_open
    from models.vllm import VLLMPolicy
    from memory.journal import JournalDB, ListDB
    from methods.construction.rollout import construct_episode
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'model', 'reference', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    config, output = load_config(args.config), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    reference, trained = tensor_files(args.reference), tensor_files(args.model)
    if reference.keys() != trained.keys():
        raise ValueError('Export parameter names differ from base')
    changed, finite, elements = [], True, 0
    for key in trained:
        with safe_open(str(trained[key]), framework='pt', device='cpu') as a, safe_open(str(reference[key]), framework='pt', device='cpu') as b:
            value, before = a.get_tensor(key), b.get_tensor(key)
            if value.shape != before.shape or value.dtype != before.dtype:
                raise ValueError('Export shape/dtype mismatch: ' + key)
            finite &= bool(torch.isfinite(value).all())
            elements += value.numel()
            if not torch.equal(value, before):
                changed.append(key)
    if not finite or not changed:
        raise ValueError('Export contains non-finite or unchanged parameters')
    policy = VLLMPolicy(args.model, config)
    db = JournalDB(ListDB())
    episode = {'sample_id': 'reload-proof', 'sessions': [{'session_id': 's1', 'time': '2026-01-01', 'turns': [
        {'turn_id': 'D1:1', 'speaker': 'Alice', 'text': 'I live in Paris and work as a teacher.'}]}]}
    operations, _ = construct_episode(episode, policy, db, config)
    valid = [op for op in operations if op['valid'] and op.get('completion_valid')]
    if not valid:
        raise ValueError('Reloaded model produced no executable construction operation')
    write_json(output / 'VERIFIED.json', {'finite': finite, 'elements': elements, 'changed_tensor_count': len(changed),
               'changed_tensor_names': changed, 'valid_tool_operations': len(valid), 'operations': operations,
               'model': str(Path(args.model).resolve()), 'reference': str(Path(args.reference).resolve())})


if __name__ == '__main__':
    main()
