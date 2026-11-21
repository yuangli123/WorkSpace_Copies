import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.joint_config import load, variant
from experiments.joint import make_runtime, corpus
from experiments.two_stage import atomic_examples
from methods.joint.rollout import collect
from methods.joint.credit import score_round
from memory.protocol import parse_calls
from scripts.make_fixture import create
from training.native import NativeUpdater
from training.separate import RoleBundle, SeparateUpdater
from evaluation.comparison import compare
from evaluation.longmemeval_prompt import get_anscheck_prompt
from tests.test_training import Tiny, rows

ROOT = Path(__file__).resolve().parents[1]


def test_disabled_roles_use_current_shared_actor(tmp_path):
    create(tmp_path / 'data')
    episodes, qs = corpus(tmp_path / 'data/train')
    for name, enabled in [('M1', {'R'}), ('M2', {'B', 'U'}), ('M3', {'B', 'U', 'R'})]:
        cfg = variant(load(ROOT / 'configs/fixture.json'), name)
        runtime, _, _, _ = make_runtime(cfg, False)
        data = collect(episodes[0], qs, runtime, 0)
        training, _ = score_round(data, runtime)
        assert {r['role'] for r in training} <= enabled
        for call in runtime.calls:
            messages, role = call['messages'], call['role']
            assert '[ROLE:' not in json.dumps(messages)
            assert role in {'B', 'U', 'R'}
        assert all(row['role'] in {'B', 'U', 'R'} for row in runtime.calls)
        if name == 'M3':
            assert not any(r['branch'] == 'local' for r in training)


def test_separate_models_update_and_resume_independently(tmp_path):
    cfg = load(ROOT / 'configs/hf-smoke.json')
    cfg['model']['gradient_checkpointing'] = False
    def build():
        policies = {role: SimpleNamespace(model=Tiny(), device=torch.device('cpu'), tokenizer=object()) for role in ('B', 'U', 'R')}
        updaters = {role: NativeUpdater(p, cfg) for role, p in policies.items()}
        return RoleBundle(policies, 'cpu', updaters)
    bundle = build()
    assert len({id(next(p.model.parameters())) for p in bundle.policies.values()}) == 3
    before = {r: copy.deepcopy(p.model.state_dict()) for r, p in bundle.policies.items()}
    selected = [dict(row, role=role) for role in ('B', 'U', 'R') for row in rows()]
    updater = SeparateUpdater(bundle)
    result = updater.update(selected)
    assert result['optimizer_updates'] == 3
    assert all(any(not torch.equal(before[r][k], v) for k, v in p.model.state_dict().items()) for r, p in bundle.policies.items())
    updater.save(tmp_path / 'ckpt', {'step': 1})
    other = SeparateUpdater(build())
    assert other.restore(tmp_path / 'ckpt') == {'step': 1}
    for role in ('B', 'U', 'R'):
        for key, value in bundle.policies[role].model.state_dict().items():
            assert torch.equal(value, other.bundle.policies[role].model.state_dict()[key])


def test_atomic_sft_keeps_completion_spans():
    from memory.branch import digest
    text = 'reasoning<tool_call>{"name":"create_raw","arguments":{"turn_ids":["t"]}}</tool_call>'
    calls = parse_calls(text)
    row = {'valid': True, 'event': 'event', 'text': text, 'messages': [{'role': 'user', 'content': 'input'}]}
    op = {'op_id': digest(['event', 0]), 'call': calls[0], 'source_turn_ids': ['t'], 'memory_refs': [], 'deleted_refs': []}
    result = atomic_examples([row], [op], 'sample', 'formation')
    assert result[0]['target_span'] == [0, len(text)]
    assert result[0]['protocol_version'] == 'paper-v2'
    assert result[0]['messages'] == row['messages']


def test_comparison_rejects_coverage_and_judge_prompts(tmp_path):
    row = {'qa_id': 'q', 'sample_id': 's', 'question': 'Q', 'answer': 'A', 'f1': .5}
    first, second = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    first.write_text(json.dumps(row) + '\n')
    second.write_text(json.dumps(dict(row, qa_id='other')) + '\n')
    with pytest.raises(ValueError, match='coverage'):
        compare({'a': first, 'b': second}, bootstrap=10)
    text = get_anscheck_prompt('knowledge-update', 'Q', 'A', 'R')
    assert 'previous information' in text
    assert 'unanswerable' in get_anscheck_prompt('multi-session', 'Q', 'A', 'R', True)
