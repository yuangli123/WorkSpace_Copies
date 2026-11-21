import json
from pathlib import Path

import torch

from experiments.io import read_json, write_json
from experiments.joint import checkpoint, restore
from training.checkpoints import CheckpointManager, verify_checkpoint


class ShardedUpdater:
    config = {'runtime': {'checkpoint_retention': 'latest_best'}}

    def save(self, path, state):
        (path / 'actor/huggingface').mkdir(parents=True)
        for rank in range(2):
            for kind in ('model', 'optim', 'extra_state'):
                torch.save({'step': state['step']}, path / ('actor/%s_world_size_2_rank_%d.pt' % (kind, rank)))
        write_json(path / 'actor/huggingface/config.json', {})
        write_json(path / 'state.json', dict(state, world_size=2))
        (path / 'COMPLETE').write_text('ok')

    def restore(self, path):
        return read_json(path / 'state.json')


def test_joint_retention_preserves_latest_and_best_without_other_run_changes(tmp_path):
    output = tmp_path / 'run'
    unrelated = tmp_path / 'another-run/model_world_size_2_rank_0.pt'
    unrelated.parent.mkdir()
    unrelated.write_bytes(b'untouched')
    updater = ShardedUpdater()
    initial = checkpoint(updater, output, {'step': 0})
    best = CheckpointManager(output, updater, updater.config).select(initial, {'f1': .4})
    latest = checkpoint(updater, output, {'step': 10})
    assert (initial / 'RETIRED_COMPLETE').exists()
    assert not list(initial.rglob('model_*.pt')) and not list(initial.rglob('optim_*.pt'))
    assert verify_checkpoint(best['path'], optimizer=False)['step'] == 0
    assert restore(updater, output)['step'] == 10
    selected = CheckpointManager(output, updater, updater.config).select(latest, {'f1': .5})
    assert verify_checkpoint(selected['path'], optimizer=False)['step'] == 10
    assert not list(Path(best['path']).rglob('model_*.pt'))
    assert unrelated.read_bytes() == b'untouched'


def test_failed_save_never_retires_latest(tmp_path):
    updater = ShardedUpdater()
    initial = checkpoint(updater, tmp_path, {'step': 0})
    def failed(path, state):
        raise OSError('simulated disk failure')
    updater.save = failed
    import pytest
    with pytest.raises(OSError, match='disk failure'):
        checkpoint(updater, tmp_path, {'step': 10})
    assert (initial / 'COMPLETE').exists() and restore(updater, tmp_path)['step'] == 0


def test_retention_through_baseline_training_and_validation(tmp_path, monkeypatch):
    import copy
    from types import MethodType
    from experiments.joint import run, no_phase, checkpoint_path
    from experiments.joint_config import load
    from memory.branch import Ranker
    from models.joint import Runtime
    from scripts.make_fixture import create
    from tests.test_pipeline import MicroPolicy
    from tests.test_training import Tiny
    from training.native import NativeUpdater
    root = Path(__file__).resolve().parents[1]
    create(tmp_path/'data')
    config = load(root/'configs/fixture.json')
    config['backend'] = 'verl'
    config['memory'].update(backend='lexical_test', embedding_model=str(tmp_path/'embedding'))
    config['models']['base'] = str(tmp_path/'base')
    config['model'].update(device='cpu', dtype='float32')
    config['retrieval'].update(steps=1)
    config['runtime'].update(checkpoint_retention='latest_best', save_every=1, validate_every=1, monitor_interval=0)
    for name in ['base','embedding']:
        (tmp_path/name).mkdir(); (tmp_path/name/'config.json').write_text('{}')
    torch.manual_seed(9)
    weights = copy.deepcopy(Tiny().state_dict())
    def factory(cfg, training):
        policy = MicroPolicy(weights)
        updater = NativeUpdater(policy, cfg)
        updater.save = MethodType(ShardedUpdater.save, updater)
        return Runtime(policy, cfg, ranker=Ranker(fixture=True)), None, no_phase, updater
    monkeypatch.setattr('experiments.joint.make_runtime', factory)
    monkeypatch.setattr('experiments.joint.common_contract', lambda cfg: {'fixture_only': True})
    output = tmp_path/'run'
    run(config, tmp_path/'data/train', output, valid_data=tmp_path/'data/valid')
    assert read_json(output/'TRAINING_COMPLETE.json')['step'] == 1
    assert read_json(output/'BASELINE.json')['step'] == 0
    selected = checkpoint_path(output, 'best')
    assert read_json(selected/'state.json')['model_only']
    receipt = read_json(selected/'RECEIPT.json')
    from experiments.io import sha256
    assert all(sha256(selected/name) == value for name,value in receipt.items())
    assert read_json(output/'BEST.json')['validation']
