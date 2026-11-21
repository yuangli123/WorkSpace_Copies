import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from experiments.io import configure_storage, read_jsonl, write_jsonl
from experiments.storage import DiskBudget, StorageBudgetExceeded, tree_bytes


def test_streamed_compressed_traces_preserve_all_fields(tmp_path):
    rows = [{'prompt_ids': list(range(10000)), 'text': '完整轨迹' * 1000, 'reward': -1.5}, {'advantage': 2.0}]
    path = tmp_path / 'traces.jsonl.gz'
    write_jsonl(path, iter(rows))
    assert read_jsonl(path) == rows
    assert gzip.open(path, 'rt', encoding='utf-8').read().count('\n') == 2
    assert not path.with_suffix('.gz.tmp').exists()


def test_budget_rejects_peak_before_write_and_keeps_checkpoint(tmp_path):
    checkpoint = tmp_path / 'COMPLETE'
    checkpoint.write_text('valid')
    config = {'allowed_root': str(tmp_path), 'roots': [str(tmp_path)],
              'ledger': str(tmp_path / 'ledger.json'), 'limit_bytes': 1_000_000,
              'warning_bytes': 800_000, 'runtime_reserve_bytes': 0}
    budget = DiskBudget(config)
    with pytest.raises(StorageBudgetExceeded):
        budget.check(1_000_000, 'new checkpoint')
    assert checkpoint.read_text() == 'valid'
    assert json.loads((tmp_path / 'ledger.json').read_text())['reason'] == 'new checkpoint'
    configure_storage({})


def test_overlapping_budget_roots_are_not_double_counted(tmp_path):
    child = tmp_path / 'child'
    child.mkdir()
    (child / 'file').write_bytes(b'x' * 2000)
    seen = set()
    assert tree_bytes(child, seen) > 0
    assert tree_bytes(child, seen) == 0


class CheckpointUpdater:
    def save(self, path, state):
        from experiments.io import write_json
        (path / 'actor/huggingface').mkdir(parents=True, exist_ok=True)
        for rank in range(4):
            for kind in ('model', 'optim', 'extra_state'):
                torch.save({'value': torch.tensor([state['step']])}, path / f'actor/{kind}_world_size_4_rank_{rank}.pt')
        write_json(path / 'actor/huggingface/config.json', {'model_type': 'test'})
        write_json(path / 'state.json', dict(state, world_size=4))
        (path / 'COMPLETE').write_text('ok')


def test_checkpoint_rotation_preserves_latest_and_best_is_model_only(tmp_path):
    from training.checkpoints import CheckpointManager, verify_checkpoint
    manager = CheckpointManager(tmp_path, CheckpointUpdater(), {})
    first = manager.save({'step': 5, 'question_cursor': 160, 'order': [0]})
    best = manager.select(first, {'f1': 0.5})
    second = manager.save({'step': 10, 'question_cursor': 320, 'order': [0]})
    assert not (first / 'COMPLETE').exists()
    assert not (first / 'actor/model_world_size_4_rank_0.pt').exists()
    assert (first / 'state.json').exists()
    assert verify_checkpoint(second)['question_cursor'] == 320
    assert manager.select(second, {'f1': 0.4}) == best
    assert verify_checkpoint(best['path'], optimizer=False)['model_only']
    with pytest.raises(ValueError, match='Model-only'):
        verify_checkpoint(best['path'])


def test_failed_checkpoint_does_not_rotate_recoverable_previous(tmp_path):
    from training.checkpoints import CheckpointManager, verify_checkpoint
    from experiments.io import read_json
    manager = CheckpointManager(tmp_path, CheckpointUpdater(), {})
    previous = manager.save({'step': 5, 'question_cursor': 160})
    def fail(path, state):
        path.mkdir(parents=True)
        (path / 'partial').write_text('incomplete')
        raise OSError('disk write failed')
    manager.updater.save = fail
    with pytest.raises(OSError):
        manager.save({'step': 10, 'question_cursor': 320})
    assert read_json(tmp_path / 'LATEST.json')['step'] == 5
    assert verify_checkpoint(previous)['step'] == 5


def test_corrupt_checkpoint_is_rejected(tmp_path):
    from training.checkpoints import CheckpointManager, verify_checkpoint
    manager = CheckpointManager(tmp_path, CheckpointUpdater(), {})
    checkpoint = manager.save({'step': 5, 'question_cursor': 160})
    file = checkpoint / 'actor/model_world_size_4_rank_0.pt'
    with file.open('r+b') as stream:
        stream.seek(100)
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='integrity failure'):
        verify_checkpoint(checkpoint)


def test_restore_comparison_checks_optimizer_and_rng_values():
    import numpy as np
    from experiments.runtime import compare_state
    left = {'step': torch.tensor(5.0), 'moment': torch.tensor([0.25, -0.5]), 'rng': np.arange(8)}
    right = {'step': left['step'].clone(), 'moment': left['moment'].clone(), 'rng': left['rng'].copy()}
    assert compare_state(left, right) == 3
    right['moment'][1] = 0
    with pytest.raises(ValueError, match='moment'):
        compare_state(left, right)


def test_managed_global_batch32_has_one_update_and_independent_validation(tmp_path, monkeypatch):
    from experiments import workflow
    from experiments.io import load_config, read_json, write_json
    cfg = load_config('configs/memt4b_4gpu_100gb.yaml')
    cfg.pop('storage')
    cfg['retrieval'].update(steps=2, save_every=1, validate_every=2, trees=1, pivots=1)
    cfg['runtime'].update(restore_probe_step=99, monitor_interval=0, stop_file=str(tmp_path / 'STOP'))
    questions = [{'qa_id': str(i), 'sample_id': 'train', 'question': 'city?', 'category': 1,
                  'answer': 'Paris', 'evidence': []} for i in range(32)]
    class Environment:
        def prompt(self, *args):
            return [{'role': 'user', 'content': 'city?'}]
        def step(self, text, seen):
            return True, True, 'Paris', '', [], seen
    class Policy:
        def complete_batch(self, messages, seeds, **kwargs):
            from memory.protocol import Completion
            return [Completion('Paris', [1], [2]) for _ in messages]
    class Updater(CheckpointUpdater):
        def __init__(self):
            self.updates = 0
        def update(self, rows):
            self.updates += 1
            assert sum(row['weight'] for row in rows) == pytest.approx(1)
            return {'actor/loss': 0.0, 'actor/kl': 0.0, 'actor/grad_norm': 1.0}
        def export(self, path, checkpoint):
            write_json(path / 'COMPLETE.json', {'checkpoint': str(checkpoint)})
    memories = []
    for split in ('train', 'valid'):
        folder = tmp_path / split
        write_json(folder / 'index.json', {})
        write_jsonl(folder / 'questions.jsonl', questions)
        memories.append(SimpleNamespace(path=folder, partition={'split': split}, questions=questions,
                                        environment=lambda *args: Environment()))
    validations = []
    def evaluate(config, memory, policy, output):
        validations.append(memory.partition['split'])
        return {'count': 32, 'f1': 0.5}
    monkeypatch.setattr(workflow, 'evaluate', evaluate)
    updater = Updater()
    output = tmp_path / 'run'
    workflow.train_retrieval(cfg, memories[0], Policy(), updater, output, memories[1])
    assert updater.updates == 2 and validations == ['valid']
    assert read_json(output / 'LATEST.json')['step'] == 2
    assert read_json(output / 'TRAINING_COMPLETE.json')['question_cursor'] == 64
    assert len(read_jsonl(output / 'rollouts/step-000002.jsonl.gz')) == 32
    assert read_json(output / 'metrics/step-000002.json')['global_query_batch'] == 32
    stopping = Updater()
    def update_then_stop(rows):
        metrics = Updater.update(stopping, rows)
        (tmp_path / 'STOP').write_text('stop at next safe boundary')
        return metrics
    stopping.update = update_then_stop
    stopped_output = tmp_path / 'stopped-run'
    workflow.train_retrieval(cfg, memories[0], Policy(), stopping, stopped_output, memories[1])
    assert stopping.updates == 1 and validations == ['valid']
    assert read_json(stopped_output / 'LATEST.json')['step'] == 1
    assert read_json(stopped_output / 'STOPPED.json')['question_cursor'] == 32
    assert not (stopped_output / 'TRAINING_COMPLETE.json').exists()


def test_stop_string_without_eos_masks_padding_and_honors_fallback():
    pytest.importorskip('vllm')
    from training.verl_backend import enable_verl
    enable_verl()
    from vllm import SamplingParams
    from verl import DataProto
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    rollout = object.__new__(vLLMRollout)
    rollout.config = SimpleNamespace(response_length=5, n=1, free_cache_engine=False)
    rollout.pad_token_id = 0
    rollout.sampling_params = SamplingParams(max_tokens=5, stop=['</tool_call>'])
    observed = {}
    def generate(**kwargs):
        observed['stop'] = list(kwargs['sampling_params'].stop)
        return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[2, 3])]),
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[4, 5, 1])])]
    rollout.inference_engine = SimpleNamespace(generate=generate)
    data = DataProto.from_dict(tensors={'input_ids': torch.tensor([[0, 2], [2, 3]]),
                                        'attention_mask': torch.tensor([[0, 1], [1, 1]]),
                                        'position_ids': torch.tensor([[0, 0], [0, 1]])},
                               meta_info={'eos_token_id': 1, 'do_sample': False,
                                          'sampling_overrides': {'stop': [], 'temperature': 0}})
    result = rollout.generate_sequences(data)
    assert result.batch['attention_mask'][:, 2:].tolist() == [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]]
    assert observed['stop'] == []
    assert rollout.sampling_params.stop == ['</tool_call>']
