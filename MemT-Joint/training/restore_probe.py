"""Actual load/save/tensor comparison; only trusted checkpoints from this project."""
from pathlib import Path

from experiments.io import checked_output, read_json, write_json, sha256
from experiments.joint import restore, checkpoint_path


def verify(run, output):
    import inspect
    import torch
    from experiments.runtime import compare_state
    run, output = Path(run), checked_output(output)
    manifest = read_json(run / 'manifest.json')
    root = Path(__file__).resolve().parents[1]
    for relative, expected in manifest['code'].items():
        file = (root / relative).resolve()
        if root not in file.parents or sha256(file) != expected:
            raise ValueError('Restore probe requires the original source version')
    config = manifest['config']
    for name, hashes in manifest['models'].items():
        location = Path(config['models']['base'] if name == 'base' else config['memory']['embedding_model'])
        for relative, expected in hashes.items():
            if sha256(location / relative) != expected:
                raise ValueError('Base/embedding identity changed before restore probe')
    if config['backend'] == 'verl':
        from training.verl_backend import VerlBackend
        updater = VerlBackend(config['models']['base'], config)
    elif config['backend'] == 'hf':
        if config['experiment'].get('variant') == 'M5':
            from training.separate import create
            _, updater = create(config, True)
        else:
            from models.hf import HFPolicy
            from training.native import NativeUpdater
            updater = NativeUpdater(HFPolicy(config['models']['base'], config), config)
    else:
        raise ValueError('A fixture is not a resumable training checkpoint')
    original = checkpoint_path(run)
    state = restore(updater, run)
    probe = output / 'resaved'
    updater.save(probe, state)
    files = list(original.rglob('*.pt'))
    if not files:
        raise ValueError('No trusted state shards to compare')
    kwargs = {'map_location': 'cpu', 'weights_only': False}
    if 'mmap' in inspect.signature(torch.load).parameters:
        kwargs['mmap'] = True
    counts = {}
    for source in files:
        relative = source.relative_to(original)
        left = torch.load(source, **kwargs)
        right = torch.load(probe / relative, **kwargs)
        counts[str(relative)] = compare_state(left, right)
        del left, right
    if read_json(original / 'state.json') != read_json(probe / 'state.json'):
        raise ValueError('Restored controller state changed')
    write_json(output / 'RESTORE_VERIFIED.json', {'step': state['step'], 'backend': config['backend'], 'compared': counts,
        'model_optimizer_scheduler_rng_roundtrip_equal': True,
        'scope': 'load -> resave state equality, not a claim of GPU continuous-training bitwise equivalence'})
    return counts
