import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.joint_config import load, validate
from experiments.joint import no_phase
from experiments.io import write_json, write_jsonl, sha256
from experiments.window import load_bundle, step_deadline, parse_response
from memory.branch import State, Ranker, digest
from models.joint import Runtime, FixturePolicy, SharedAnswer, AnswerQueue
from methods.joint.rollout import blocks, formation, evolution
from methods.joint.window import collect_window
from methods.joint.credit import score_round

ROOT = Path(__file__).resolve().parents[1]


def test_offline_json_repairs_only_unmatched_closers():
    assert parse_response('{"accept": true}\n}') == {'accept': True}
    assert parse_response('```json\n[{"accept": false}]\n```') == [{'accept': False}]
    for text in ('{"accept":true}\n{"accept":false}', '{"accept":true} explanation'):
        with pytest.raises(ValueError):
            parse_response(text)


def example():
    config = load(ROOT / 'configs/window-a100x2-pilot.json')
    config['backend'] = 'fixture'
    policy = FixturePolicy()
    runtime = Runtime(policy, config, ranker=Ranker(fixture=True), fixed=SharedAnswer(policy))
    episode = {'sample_id': 'train-only', 'sessions': []}
    for si in range(6):
        episode['sessions'].append({'session_id': 's%d' % si, 'time': '2026-01-%02d' % (si+1),
            'turns': [{'turn_id': 's%d:t%d' % (si, ti), 'speaker': 'user',
                       'text': 'residence=' + ('FUTURE_CITY' if si == 5 else ['Rome', 'Paris', 'London', 'Tokyo', 'Berlin'][si])}
                      for ti in range(8)]})
    sequence = list(blocks(episode, 4))
    initial = State(episode['sample_id'], runtime.ranker)
    for bi, block in enumerate(sequence[:2]):
        initial.observe(block.session, block.turns, block.position, runtime.count)
        initial, pending, _, _ = formation(initial, block, runtime, 'offline/%d/B' % bi, greedy=True)
        initial, _, _, _ = evolution(initial, pending, block, runtime, 'offline/%d/U' % bi, greedy=True)
    qs = [{'sample_id': 'train-only', 'qa_id': 'q%d' % qi, 'question': 'Current residence? %d' % qi,
           'answer': 'GOLD_SECRET', 'evidence': ['s4:t1' if qi < 2 else 's0:t1'], 'eval_session': 4, 'reviewed': True} for qi in range(4)]
    spec = {'sample_id': 'train-only', 'eval_session': 4, 'start': 2, 'end': 10}
    runtime.calls = []
    return config, runtime, episode, initial, qs, spec


def test_window_paths_credit_isolation_and_shared_answer():
    cfg, runtime, episode, initial, questions, spec = example()
    before = initial.fingerprint()
    result = collect_window(episode, questions, initial, spec, runtime, 0)
    rows, _ = score_round(result, runtime)
    assert initial.fingerprint() == before
    assert {r['role'] for r in rows} == {'B', 'U', 'R'}
    for role in ('B', 'U', 'R'):
        assert sum(r['weight'] for r in rows if r['role'] == role) == pytest.approx(1/3)
    assert {'global', 'local'} == {r['branch'] for r in rows if r['role'] == 'U'}
    assert all(r['position'][0] >= 1 for r in rows if r['role'] in ('B', 'U'))
    assert all('/continuation-' not in r['event'] for r in rows)
    assert all('offline/' not in r['event'] for r in rows)
    assert len(result['global']) == 2 and len(result['local_B'][0]) == len(result['local_U'][0]) == 2
    assert len(result['local_R']) == 4 and all(len(r['trees']) == 3 for r in result['local_R'])
    assert all('FUTURE_CITY' not in json.dumps(r['messages']) and 'GOLD_SECRET' not in json.dumps(r['messages'])
               for r in runtime.calls)
    assert all('GOLD_SECRET' not in json.dumps(row['messages']) for row in runtime.calls)
    assert all(r['evaluation_horizon'] == 'window_end' for group in (result['local_B'], result['local_U'])
               for r in group[0])
    assert all(len(p['blocks']) == 8 for p in result['global'])
    final_hashes = {State.from_dict(p['snapshots'][-1]['state'], runtime.ranker).fingerprint() for p in result['global']}
    assert {r['state_hash'] for r in result['local_R']} <= final_hashes
    assert any(node.answer for record in result['local_R'] for tree in record['trees'] for node in tree[1:])


def test_answer_follows_current_policy_and_cache_version_excludes_labels():
    class Current:
        value = 'before'
        calls = 0
        def complete_batch(self, messages, seeds, **kwargs):
            from memory.protocol import Completion
            self.calls += 1
            assert kwargs == {'temperature': 0, 'stop_at_tool': False}
            assert all('GOLD' not in json.dumps(m) for m in messages)
            return [Completion(self.value, [1], [2]) for _ in messages]
    policy = Current()
    answerer = SharedAnswer(policy)
    q = {'question': 'where?', 'answer': 'GOLD', 'evidence': ['GOLD']}
    first = AnswerQueue(0)
    key = first.submit(q, ['shown'], 'source_backed')
    assert first.submit(dict(q, answer='DIFFERENT'), ['shown'], 'source_backed') == key
    assert AnswerQueue(0, {'max_new_tokens': 512}).submit(q, ['shown'], 'source_backed') != key
    first.resolve(answerer, no_phase)
    assert first.text(key) == 'before'
    policy.value = 'after'
    second = AnswerQueue(1)
    new_key = second.submit(q, ['shown'], 'source_backed')
    second.resolve(answerer, no_phase)
    assert new_key != key and second.text(new_key) == 'after' and policy.calls == 2
    assert first.text(key) == 'before'


def test_generation_session_retains_worker_weights_until_end():
    # Execute the actual worker methods with CPU stand-ins for collective/GPU operations.
    file = ROOT / 'third_party/Tree-GRPO/verl/workers/fsdp_workers.py'
    tree = ast.parse(file.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActorRolloutRefWorker')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('begin_generation_session', 'end_generation_session', 'generate_sequences')]
    for n in methods:
        n.decorator_list = []
        n.returns = None
        for a in n.args.args:
            a.annotation = None
    stub = ast.Module(body=[ast.ClassDef(name='Worker', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    ast.fix_missing_locations(stub)
    from contextlib import nullcontext
    events = []
    class Manager:
        def __enter__(self): events.append('sync')
        def __exit__(self, *_): events.append('sleep')
        def preprocess_data(self, p): return p
        def postprocess_data(self, p): return p
    namespace = {'nullcontext': nullcontext, 'logger': None, 'log_gpu_memory_usage': lambda *_a, **_k: None,
        'load_fsdp_param_and_grad': lambda **_k: events.append('load'),
        'offload_fsdp_param_and_grad': lambda **_k: events.append('offload'),
        'torch': SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0, empty_cache=lambda: None))}
    exec(compile(stub, str(file), 'exec'), namespace)
    worker = namespace['Worker']()
    worker._is_rollout = worker._is_offload_param = True
    worker._is_offload_grad = False
    worker.actor_module_fsdp = None
    worker.rollout_sharding_manager = Manager()
    class Data:
        def __init__(self): self.batch, self.meta_info = self, {}
        def to(self, *_): return self
        def cuda(self): return self
    worker.rollout = SimpleNamespace(generate_sequences=lambda prompts: prompts)
    worker.generation_config = None
    worker.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    worker.begin_generation_session()
    for _ in range(3): worker.generate_sequences(Data())
    worker.end_generation_session()
    assert events == ['load', 'sync', 'sleep', 'offload']
    events.clear()
    worker.generate_sequences(Data())
    assert events == ['load', 'sync', 'sleep', 'offload']


def test_driver_session_exception_cleanup_and_update_guard():
    from training.verl_backend import VerlBackend
    backend = VerlBackend.__new__(VerlBackend)
    events = []
    backend.actor = SimpleNamespace(begin_generation_session=lambda: events.append('begin'),
                                    end_generation_session=lambda: events.append('end'))
    backend.session_depth, backend.generation_sessions, backend.sync_seconds = 0, 0, 0.
    with pytest.raises(RuntimeError):
        with backend.generation_session():
            with backend.generation_session():
                with pytest.raises(AssertionError, match='Generation must sleep'):
                    backend.update([])
                raise RuntimeError('sampling failed')
    assert events == ['begin', 'end'] and not backend.session_depth and backend.generation_sessions == 1


def test_dp_balance_preserves_samples_loss_and_gradient():
    import torch
    from training.verl_backend import balanced_rows
    from methods.joint.loss import joint_loss
    rows = [{'prompt_ids': [1]*(n-1), 'response_ids': [2], 'weight': 1/6,
             'advantage': (i-2)/3, 'event': str(i)} for i, n in enumerate((100, 95, 90, 85, 10, 8))]
    ordered, loads = balanced_rows(rows, 2)
    assert max(loads)-min(loads) <= 2
    assert {id(r) for r, real in ordered if real} == {id(r) for r in rows}
    assert all(real for _, real in ordered)
    def objective(values):
        parameter = torch.tensor(.1, dtype=torch.float64, requires_grad=True)
        current = torch.stack([parameter*(int(r['event'])+1) for r in values])[:, None]
        old, ref = torch.zeros_like(current), torch.ones_like(current)*.02
        advantage = torch.tensor([r['advantage'] for r in values], dtype=torch.float64)
        weight = torch.tensor([r['weight'] for r in values], dtype=torch.float64)
        loss, _ = joint_loss(current, old, ref, advantage, torch.ones_like(current), weight)
        loss.backward()
        return loss.item(), parameter.grad.item()
    assert objective(rows) == pytest.approx(objective([r for r, _ in ordered]), abs=1e-10)
    padded, loads = balanced_rows(rows[:5], 2)
    assert len(padded) == 6 and sum(real for _, real in padded) == 5
    assert all(r['weight'] == 0 for r, real in padded if not real)
    assert [len(padded[:3]), len(padded[3:])] == [3, 3]


def test_window_protocol_is_disabled_for_end_to_end_runs():
    cfg = load(ROOT / 'configs/window-a100x2-pilot.json')
    with pytest.raises(ValueError, match='empty-state end-to-end'):
        validate(cfg, training=True)


def test_deadline_fails_instead_of_finishing_a_partial_step():
    import signal
    if not hasattr(signal, 'setitimer'):
        pytest.skip('Linux-only wall clock deadline')
    import time
    with pytest.raises(TimeoutError):
        with step_deadline(.01): time.sleep(.1)


def write_bundle(tmp_path, episode, initial, questions, spec, models=None):
    data, root = tmp_path / 'data/train', tmp_path / 'bundle'
    write_jsonl(data / 'episodes.jsonl', [episode])
    write_jsonl(data / 'questions.jsonl', [])
    write_json(data / 'partition.json', {'split': 'train'})
    prefix = dict(episode, sessions=episode['sessions'][:spec['eval_session']+1])
    blind = [{'answer': q['answer'], 'evidence': q['evidence'], 'answerable': True} for q in questions]
    verdicts = [{'accept': True, 'reason': 'synthetic CPU test only'} for _ in questions]
    def messages(value): return [{'role': 'user', 'content': json.dumps(value)}]
    write_json(root / 'blind.json', {'messages': messages({'prefix': prefix, 'questions': [q['question'] for q in questions],
        'cutoff_time': prefix['sessions'][-1]['time']}), 'response': json.dumps(blind)})
    write_json(root / 'audit.json', {'messages': messages({'prefix': prefix,
        'candidates': [dict(question=q['question'], answer=q['answer'], evidence=q['evidence'], blind=b)
                       for q, b in zip(questions, blind)]}), 'response': json.dumps(verdicts)})
    for i, q in enumerate(questions):
        q['source_cutoff'] = {'prefix_sha256': digest(prefix)}
        q['review'] = {'blind_file': 'blind.json', 'blind_sha256': sha256(root / 'blind.json'),
            'audit_file': 'audit.json', 'audit_sha256': sha256(root / 'audit.json'), 'index': i,
            'blind_answer': blind[i], 'verdict': verdicts[i]}
    write_jsonl(root / 'probes.jsonl', questions)
    write_json(root / 'state.json', initial.to_dict())
    spec = dict(spec, snapshot='state.json', snapshot_sha256=sha256(root / 'state.json'))
    bundle = {'version': 'memt-window-v2', 'split': 'train', 'windows': [spec], 'probes': 'probes.jsonl',
        'probes_sha256': sha256(root / 'probes.jsonl'), 'episodes_sha256': sha256(data / 'episodes.jsonl'),
        'frozen_prefix_model': models or {}}
    write_json(root / 'bundle.json', bundle)
    write_json(root / 'ASSISTANT_AUDIT.json', {'bundle_sha256': sha256(root / 'bundle.json'), 'source_checked': True,
        'accepted_qa_ids': [q['qa_id'] for q in questions], 'scope': 'SYNTHETIC TEST ONLY'})
    return data, root


def test_bundle_binds_qa_snapshot_review_and_cutoff(tmp_path):
    _, _, episode, initial, qs, spec = example()
    data, root = write_bundle(tmp_path, episode, initial, qs, spec)
    windows, probes, files = load_bundle(root / 'bundle.json', [episode], data, {})
    assert len(windows) == 1 and len(probes) == 4 and len(files) == 6
    # Rehashing a modified label does not make the original blind review support it.
    qs[0]['answer'] = 'ALTERED_AFTER_REVIEW'
    write_jsonl(root / 'probes.jsonl', qs)
    bundle = json.loads((root / 'bundle.json').read_text())
    bundle['probes_sha256'] = sha256(root / 'probes.jsonl')
    write_json(root / 'bundle.json', bundle)
    audit = json.loads((root / 'ASSISTANT_AUDIT.json').read_text())
    audit['bundle_sha256'] = sha256(root / 'bundle.json')
    write_json(root / 'ASSISTANT_AUDIT.json', audit)
    with pytest.raises(ValueError, match='differs from the independently reviewed'):
        load_bundle(root / 'bundle.json', [episode], data, {})


def test_single_item_audit_receipt_and_unreviewed_refusal(tmp_path):
    _, _, episode, initial, qs, spec = example()
    data, root = write_bundle(tmp_path, episode, initial, qs, spec)
    receipt = json.loads((root / 'audit.json').read_text())
    payload = json.loads(receipt['messages'][-1]['content'])
    payload['candidates'] = payload['candidates'][:1]
    receipt['messages'][-1]['content'] = json.dumps(payload)
    receipt['response'] = json.dumps(qs[0]['review']['verdict'])
    write_json(root / 'single.json', receipt)
    qs[0]['review'].update(audit_file='single.json', audit_sha256=sha256(root / 'single.json'), audit_index=0)
    def seal():
        write_jsonl(root / 'probes.jsonl', qs)
        bundle = json.loads((root / 'bundle.json').read_text())
        bundle['probes_sha256'] = sha256(root / 'probes.jsonl')
        write_json(root / 'bundle.json', bundle)
        audit = json.loads((root / 'ASSISTANT_AUDIT.json').read_text())
        audit['bundle_sha256'] = sha256(root / 'bundle.json')
        write_json(root / 'ASSISTANT_AUDIT.json', audit)
    seal()
    assert len(load_bundle(root / 'bundle.json', [episode], data)[1]) == 4
    qs[0]['reviewed'] = False
    seal()
    with pytest.raises(ValueError, match='Missing reviewed QA'):
        load_bundle(root / 'bundle.json', [episode], data)


def test_missing_history_qa_refused_before_model_initialization(tmp_path):
    _, _, episode, initial, qs, spec = example()
    for q in qs:
        q['evidence'] = ['s4:t1']
    data, root = write_bundle(tmp_path, episode, initial, qs, spec)
    with pytest.raises(ValueError, match='prefix-retention QA before initialization'):
        load_bundle(root / 'bundle.json', [episode], data)


@pytest.mark.skip(reason='Window prefix protocol is disabled; use empty-state end-to-end runner')
def test_window_runner_real_cpu_updates_and_answer_versions(tmp_path, monkeypatch):
    import torch
    from tests.test_pipeline import MicroPolicy
    from tests.test_training import Tiny
    from training.native import NativeUpdater
    from experiments.joint import run
    cfg, _, episode, initial, qs, spec = example()
    cfg['backend'] = 'hf'
    cfg['model'].update(device='cpu', dtype='float32', gradient_checkpointing=False)
    cfg['runtime']['monitor_interval'] = 0
    cfg['retrieval']['steps'] = 2
    # CPU tests check the runner; the Linux 1800-second guard is tested separately.
    cfg['window']['step_seconds'] = 1800
    base, embedding = tmp_path / 'base', tmp_path / 'embedding'
    base.mkdir(); embedding.mkdir()
    torch.manual_seed(9)
    weights = copy.deepcopy(Tiny().state_dict())
    torch.save(weights, base / 'model.bin')
    (embedding / 'config.json').write_text('{}')
    cfg['models']['base'], cfg['memory']['embedding_model'] = str(base), str(embedding)
    data, root = write_bundle(tmp_path, episode, initial, qs, spec, {'model.bin': sha256(base / 'model.bin')})
    cfg['window']['bundle'] = str(root / 'bundle.json')
    factory_policies = []
    class Current(MicroPolicy):
        def complete(self, messages, **kwargs):
            if '[ROLE:Answer]' in messages[0]['content']:
                from memory.protocol import Completion
                with torch.no_grad():
                    token = int(self.model(torch.tensor([[4, 1]])).logits[0, -1].argmax())
                return Completion('answer-%d' % token, [4, 1], [token])
            return super().complete(messages, **kwargs)
    def factory(config, training):
        policy = Current(weights)
        factory_policies.append(policy)
        answerer = SharedAnswer(policy)
        runtime = Runtime(policy, config, fixed=answerer, ranker=Ranker(fixture=True))
        return runtime, answerer, no_phase, NativeUpdater(policy, config)
    monkeypatch.setattr('experiments.joint.make_runtime', factory)
    # Windows has no setitimer; replace only the deadline, not sampling/scoring/updates.
    from contextlib import nullcontext
    monkeypatch.setattr('experiments.window.step_deadline', lambda _: nullcontext())
    out = tmp_path / 'run'
    run(cfg, data, out)
    assert json.loads((out / 'TRAINING_COMPLETE.json').read_text())['step'] == 2
    assert any(not torch.equal(weights[k], v) for k, v in factory_policies[0].model.state_dict().items())
    metrics = sorted((out / 'attempts').glob('*/step-*/metrics.json'))
    assert [json.loads(p.read_text())['answer_policy_version'] for p in metrics] == [0, 1]
    for p in metrics:
        assert json.loads(p.read_text())['ordinary_step_seconds_including_checkpoint'] > 0
        answers = json.loads((p.parent / 'answers.json').read_text())
        assert answers['interface'] == 'finish(answer)'


def test_production_window50_keeps_roles_and_disables_pilot_timeout():
    cfg = load(ROOT / 'configs/common-a100x2-50.json')
    validate(cfg, training=True)
    assert cfg['retrieval']['steps'] == 50 and 'window' not in cfg
    assert cfg['runtime']['save_every'] == cfg['runtime']['validate_every'] == 10
    assert set(cfg['experiment']['train_roles']) == {'B', 'U', 'R'}
    assert cfg['joint']['branch_weights']['B']['local'] > 0 and cfg['joint']['branch_weights']['U']['local'] > 0
