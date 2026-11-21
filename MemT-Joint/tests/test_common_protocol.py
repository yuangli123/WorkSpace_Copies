import json
from pathlib import Path

import pytest

from evaluation.comparison import compare
from experiments.joint import no_phase
from experiments.joint_config import contract, load, validate, variant
from memory.branch import State, builder
from memory.protocol import Completion
from methods.joint.credit import score_record
from methods.joint.prompts import ANSWER_RULES, retrieval_prompt, tool_schemas
from methods.joint.retrieval import sample
from models.joint import Runtime, FixturePolicy

ROOT = Path(__file__).resolve().parents[1]


def test_strategy_changes_do_not_change_inference_contract():
    config = load(ROOT / 'configs/common-a100x2-50.json')
    expected = contract(config)
    for name in ('M0', 'M1', 'M2', 'M3', 'M4', 'M5'):
        assert contract(variant(config, name)) == expected
    assert expected['prompt_tokens'] + expected['output_tokens'] == 40960
    m6 = load(ROOT / 'configs/common-m6.json')
    assert m6['backend'] == 'hf' and contract(m6) == expected
    config['retrieval']['observation_tokens'] = 2048
    with pytest.raises(ValueError, match='agreed common protocol'):
        validate(config)


def test_tools_and_answer_rules_are_explicit_and_share_no_gold():
    schemas = tool_schemas('source_backed', 5)
    assert len(schemas) == 6
    finish = schemas[-1]['function']['parameters']
    assert finish['required'] == ['answer'] and 'memory_refs' not in finish['properties']
    assert 'search_turns' not in json.dumps(tool_schemas('memory_only', 5))
    assert 'short phrase' in ANSWER_RULES and '15 July, 2023' in ANSWER_RULES
    assert 'last week' in ANSWER_RULES
    assert 'open-domain problem' in retrieval_prompt({'question': 'q', 'category': 3, 'benchmark': 'locomo'},
                                                    'source_backed', 5, 6)[0]['content']


class BrokenAfterSearch(FixturePolicy):
    def complete(self, messages, **kwargs):
        text = ('malformed' if '<observation>' in messages[-1]['content'] else
                '<tool_call>{"name":"search_summary","arguments":{"query":"residence"}}</tool_call>')
        return Completion(text, [1], [2])


def test_fallback_uses_visible_sources_but_never_repairs_invalid_r_reward():
    config = load(ROOT / 'configs/fixture.json')
    config['evaluation']['fallback_answer'] = True
    runtime = Runtime(BrokenAfterSearch(), config, fixed=FixturePolicy())
    state = State('s', runtime.ranker)
    turns = [{'turn_id': 'known', 'speaker': 'User', 'text': 'residence=Beijing'}]
    state.observe({'session_id': 'one', 'time': '2026-01-01'}, turns, (0, 0), runtime.count)
    completion = Completion('<tool_call>{"name":"update_summary","arguments":{"content":"residence=Beijing"}}</tool_call>', [1], [2])
    state, _, _ = builder(state, turns, completion, 'B')
    question = {'question': 'current residence?', 'answer': 'Beijing', 'qa_id': 'q'}
    record = sample(state, question, runtime, 'fallback', forest=False)
    utility = score_record(record, runtime, mot=True)
    leaf = record['trees'][0][-1]
    assert not leaf.valid and leaf.fallback and leaf.answer
    assert leaf.reward == config['reward']['invalid'] and leaf.performance == 0
    assert leaf.answer_performance >= 0 and utility <= 1
    assert set(leaf.answer_refs) <= leaf.observed.keys()
    assert all('GOLD' not in json.dumps(call['messages']) for call in runtime.calls)


def test_common_m0_generates_with_primary_policy_not_hf_answer():
    config = load(ROOT / 'configs/fixture.json')
    config['protocol'] = {'name': 'memt-common-v1'}
    config['experiment']['train_roles'] = []
    policy, answer = FixturePolicy(), FixturePolicy()
    runtime = Runtime(policy, config, fixed=answer)
    runtime.call('R', [{'role': 'system', 'content': 'search_facts'}, {'role': 'user', 'content': 'q'}], 'm0')
    assert len(policy.calls) == 1 and not answer.calls


def test_comparison_rejects_protocol_drift(tmp_path):
    paths = {}
    for name, protocol in [('one', 'contract-one'), ('two', 'contract-two')]:
        path = tmp_path / (name + '.jsonl')
        path.write_text(json.dumps({'qa_id': 'q', 'sample_id': 's', 'question': '?', 'answer': 'x', 'f1': 1., 'protocol_id': protocol})+'\n')
        paths[name] = path
    with pytest.raises(ValueError, match='Inference protocols differ'):
        compare(paths)


def test_vllm_weight_refresh_follows_native_updates_without_reloading_unchanged_weights():
    import torch
    from types import SimpleNamespace
    from models.vllm import VLLMPolicy
    owner = SimpleNamespace(model=torch.nn.Linear(2, 2), device=torch.device('cpu'))
    loaded = []
    class Model:
        def load_weights(self, items):
            loaded.append({k: v.clone() for k, v in items})
    class Engine:
        def wake_up(self):
            pass
        def sleep(self, level):
            assert level == 1
        def collective_rpc(self, method, args):
            return method(SimpleNamespace(get_model=lambda: Model()), *args)
    policy = VLLMPolicy.__new__(VLLMPolicy)
    policy.owner, policy.offload, policy.engine, policy.depth, policy.dirty = owner, None, Engine(), 0, True
    with policy.phase():
        pass
    with policy.phase():
        pass
    assert len(loaded) == 1
    with torch.no_grad():
        owner.model.weight.add_(1)
    policy.invalidate()
    with policy.phase():
        pass
    assert len(loaded) == 2 and torch.equal(loaded[-1]['weight'], owner.model.weight)


def test_response_only_projection_preserves_log_probs_and_gradients():
    import torch
    from types import SimpleNamespace
    from methods.retrieval.loss import causal_log_probs
    from tests.test_training import Tiny
    class Limited(Tiny):
        def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=0):
            logits = super().forward(input_ids).logits
            return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)
    a, b = Tiny(), Limited()
    b.load_state_dict(a.state_dict())
    left = causal_log_probs(a, [1, 2, 3], [4, 5], 'cpu')
    right = causal_log_probs(b, [1, 2, 3], [4, 5], 'cpu')
    torch.testing.assert_close(left, right)
    left.sum().backward(); right.sum().backward()
    for x, y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.grad, y.grad)


def test_packed_projection_preserves_variable_length_probabilities_and_gradients():
    import copy
    import torch
    from types import SimpleNamespace
    from methods.retrieval.loss import causal_log_probs, packed_response_log_probs
    from tests.test_training import Tiny
    class Packed(Tiny):
        def forward(self, input_ids, position_ids=None, logits_to_keep=0, **kwargs):
            hidden = self.embedding(input_ids)
            if position_ids is None:
                hidden = hidden.cumsum(1)
            else:
                starts = (position_ids[0] == 0).nonzero().flatten().tolist() + [hidden.shape[1]]
                hidden = torch.cat([hidden[:, a:b].cumsum(1) for a,b in zip(starts, starts[1:])], 1)
            selected = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            return SimpleNamespace(logits=self.head(hidden[:, selected]))
    a = Packed(); b = copy.deepcopy(a)
    expected = torch.stack([causal_log_probs(a, [1,2,3], [4,5], 'cpu', .8)[0],
                            torch.nn.functional.pad(causal_log_probs(a, [2,3], [6], 'cpu', .8)[0], (0,1))])
    batch = {'input_ids':torch.tensor([[1,2,3,4,5],[0,2,3,6,0]]),
             'attention_mask':torch.tensor([[1,1,1,1,1],[0,1,1,1,0]]),
             'position_ids':torch.tensor([[0,1,2,3,4],[0,0,1,2,0]]),
             'responses':torch.tensor([[4,5],[6,0]])}
    actual = packed_response_log_probs(b, batch, .8)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward(); expected.sum().backward()
    for x,y in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(x.grad, y.grad)


def test_frozen_answer_releases_driver_cache_after_offload_even_on_error():
    from types import SimpleNamespace
    from models.frozen import FrozenModel
    events = []
    class Model:
        def to(self, device):
            events.append('gpu')
        def cpu(self):
            events.append('cpu')
    frozen = FrozenModel.__new__(FrozenModel)
    frozen.torch = SimpleNamespace(device=lambda x:x, cuda=SimpleNamespace(
        is_available=lambda:True, empty_cache=lambda:events.append('empty')))
    frozen.actor, frozen.updater = SimpleNamespace(), None
    frozen.policy = SimpleNamespace(model=Model(), device='cpu')
    frozen.device, frozen.depth = 'cuda:0', 0
    with frozen.phase():
        pass
    assert events[-2:] == ['cpu','empty'] and frozen.depth == 0
    with pytest.raises(ValueError):
        with frozen.phase():
            raise ValueError('inference failed')
    assert events[-2:] == ['cpu','empty'] and frozen.depth == 0


def test_separate_roles_switch_one_generation_engine_without_sharing_parameters(monkeypatch):
    from types import SimpleNamespace
    from training.separate import RoleBundle
    created = []
    class Generation:
        def __init__(self, path, config, owner, offload):
            self.owner, self.dirty = owner, False
            created.append(self)
        def invalidate(self):
            self.dirty = True
    monkeypatch.setattr('models.vllm.VLLMPolicy', Generation)
    owners = {r:SimpleNamespace(model_path='base', tokenizer=None) for r in 'BUR'}
    bundle = RoleBundle(owners, 'cpu')
    bundle.common_config = {}
    generation = bundle.generation_for('B')
    for r in 'URB':
        assert bundle.generation_for(r) is generation
        assert generation.owner is owners[r] and generation.dirty
    assert len(created) == 1 and len({id(p) for p in owners.values()}) == 3


def test_generation_cleanup_is_explicit_and_idempotent(monkeypatch):
    from types import SimpleNamespace
    from models.vllm import VLLMPolicy
    events = []
    engine = SimpleNamespace(close=lambda:events.append('close'))
    policy = VLLMPolicy.__new__(VLLMPolicy)
    policy.engine = engine
    policy.close(); policy.close()
    assert events == ['close'] and policy.engine is None


def test_ray_engine_reports_cleanup_failure_and_stops_only_its_worker():
    from types import SimpleNamespace
    from models.vllm import _RayEngine
    events = []
    actor = SimpleNamespace(close=SimpleNamespace(remote=lambda:'cleanup'))
    def get(value):
        assert value == 'cleanup'
        raise ValueError('cleanup failure')
    engine = _RayEngine.__new__(_RayEngine)
    engine.actor = actor
    engine.ray = SimpleNamespace(get=get, kill=lambda handle,no_restart:events.append((handle,no_restart)))
    with pytest.raises(ValueError, match='cleanup failure'):
        engine.close()
    assert events == [(actor,True)]
