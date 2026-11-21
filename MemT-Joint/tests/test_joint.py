import copy
import json
from pathlib import Path

import pytest

from experiments.joint_config import load, validate
from experiments.joint import corpus, run, evaluate, make_runtime
from scripts.make_fixture import create
from memory.branch import State, Ranker, InvalidAction, builder, updater, updater_prompt
from memory.protocol import Completion
from methods.joint.rollout import collect
from methods.joint.credit import score_round, standardize
from methods.joint.retrieval import Node, step, sample, prompt

ROOT = Path(__file__).resolve().parents[1]


def tool(name, **arguments):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments}) + '</tool_call>'


def complete(text):
    return Completion(text, [1], [2])


@pytest.fixture
def setup(tmp_path):
    create(tmp_path / 'data')
    return load(ROOT / 'configs/fixture.json'), tmp_path / 'data'


def test_raw_and_transactions():
    state = State('s', Ranker(fixture=True))
    session = {'session_id': 's0', 'time': '2026-01-01'}
    block = [{'turn_id': 't0', 'speaker': 'User', 'text': 'residence=Beijing'}]
    state.observe(session, block, (0, 0), len)
    original = state.fingerprint()
    invalid = tool('update_summary', content='valid') + tool('create_raw', turn_ids=['future'])
    partial, _, operations = builder(state, block, complete(invalid), 'bad')
    assert partial.summary == 'valid'
    assert [op['valid'] for op in operations] == [True, False]
    assert not partial.journal.get(partial.collection('turns'))['ids']
    assert state.fingerprint() == original
    changed, candidates, _ = builder(state, block, complete(tool('create_fact', fact='residence=Beijing', start_time='', end_time='') +
                                   tool('create_raw', turn_ids=['t0'])), 'good')
    assert not state.journal.get(state.collection('turns'))['ids']
    assert changed.journal.get(changed.collection('turns'))['ids']
    messages, related = updater_prompt(changed, candidates[0], 3)
    integrated, _ = updater(changed, candidates[0], related, complete(tool('add_item', document='residence=Beijing')), 'add')
    _, related = updater_prompt(integrated, candidates[0], 3)
    key = related[0]['id']
    before = integrated.fingerprint()
    partial, operations = updater(integrated, candidates[0], related,
        complete(tool('delete_item', id=key) + tool('update_item', id=key, document='residence=Shanghai')), 'partial')
    assert [op['valid'] for op in operations] == [True, False]
    assert not partial.journal.get(partial.collection('facts'))['ids']
    assert integrated.fingerprint() == before
    assert integrated.clone().fingerprint() == before


def test_unread_submission_and_memory_only(setup):
    cfg, _ = setup
    runtime, _, _, _ = make_runtime(cfg, False)
    state = State('s', runtime.ranker)
    node = Node(1, 0, 1, [])
    with pytest.raises(InvalidAction):
        step(state, node, tool('finish', memory_refs=['forged@1']), runtime)
    step(state, node, tool('finish', answer='secret'), runtime)
    assert node.answer == 'secret' and node.stop_reason == 'finish'


def test_direct_finish_scores_observed_evidence(setup):
    cfg, _ = setup
    runtime, _, _, _ = make_runtime(cfg, False)
    runtime.mode = 'source_backed'
    node = Node(1, 0, 1, [], observed={'memory@1': {'sources': {'turn-1': 'User: secret'}}},
                answer='secret', answer_refs=['memory@1'], terminal=True, valid=True,
                stop_reason='finish')
    record = {'question': {'answer': 'secret', 'evidence': ['turn-1']}, 'trees': [[Node(0, -1, 0, []), node]]}
    from methods.joint.credit import score_record
    score_record(record, runtime)
    assert node.answer_evidence == 1.0 and node.evidence == 1.0 and node.performance == 1.0
    runtime.mode = 'memory_only'
    state = State('s', runtime.ranker)
    with pytest.raises(InvalidAction):
        step(state, node, tool('search_turns', query='x'), runtime)


def test_complete_round_no_gold_or_future_leak(setup):
    cfg, data = setup
    episodes, qs = corpus(data / 'train')
    runtime, _, _, _ = make_runtime(cfg, False)
    for q in qs:
        q['answer'] = 'GOLD_MUST_NOT_ENTER_PROMPTS'
    result = collect(episodes[0], qs, runtime, 0)
    rows, groups = score_round(result, runtime)
    assert rows and {'B', 'U', 'R'} <= {r['role'] for r in rows}
    assert all('GOLD_MUST_NOT_ENTER_PROMPTS' not in json.dumps(r['messages']) for r in runtime.calls)
    assert all('GOLD_MUST_NOT_ENTER_PROMPTS' not in json.dumps(r['messages']) for r in runtime.calls)
    for path in result['global']:
        early = next(r for r in path['evaluations'] if r['question']['eval_session'] == 0)
        assert early['eval_position'][0] == 0
        for node in early['trees'][0][1:]:
            assert 'Shanghai' not in json.dumps(node.messages)
            assert 's1:u1' not in node.displayed
    # Global retrieval never receives a policy-training row.
    assert all(not (r['role'] == 'R' and '/global-' in r['event']) for r in rows)
    for record in result['local_R']:
        nodes = [n for t in record['trees'] for n in t[1:]]
        assert len({n.rows[0]['event'] for n in nodes}) == len(nodes)
    assert standardize([4., 4.]) == [0., 0.]


def test_fixture_cli_flow_and_no_fake_training(setup, tmp_path):
    cfg, data = setup
    run(cfg, data / 'train', tmp_path / 'smoke', smoke=True)
    marker = json.loads((tmp_path / 'smoke/SMOKE_COMPLETE.json').read_text())
    assert marker['optimizer_updates'] == 0 and marker['fixture_only']
    assert not (tmp_path / 'smoke/TRAINING_COMPLETE.json').exists()
    with pytest.raises(ValueError, match='not on-policy'):
        validate(cfg, training=True)
    with pytest.raises(ValueError, match='train partition'):
        run(cfg, data / 'test', tmp_path / 'bad-split', smoke=True)
    summary = evaluate(cfg, data / 'test', tmp_path / 'eval')
    assert summary['count'] == 2 and summary['accuracy'] is None
    trajectories = json.loads(__import__('gzip').open(tmp_path / 'eval/trajectories.json.gz', 'rt').read())
    assert 'Tokyo' not in json.dumps(trajectories[0]['trees'][0][1]['messages'])


def test_probe_rejects_future_source(setup, tmp_path):
    _, data = setup
    probe = {'sample_id': 'fixture-train', 'qa_id': 'bad', 'question': '?', 'answer': 'x',
             'evidence': ['s1:u1'], 'eval_session': 0, 'reviewed': True}
    path = tmp_path / 'probes.jsonl'
    path.write_text(json.dumps(probe) + '\n')
    with pytest.raises(ValueError, match='future'):
        corpus(data / 'train', path)


def test_terminal_qa_is_not_a_mid_session_local_probe(setup):
    cfg, _ = setup
    cfg['joint']['local_evaluation'] = 'current_only'
    cfg['joint']['local_anchors'] = 10
    runtime, _, _, _ = make_runtime(cfg, False)
    episode = {'sample_id': 'one-session', 'sessions': [{'session_id': 's', 'time': '2026-01-01', 'turns': [
        {'turn_id': 'first', 'speaker': 'User', 'text': 'residence=Beijing'},
        {'turn_id': 'last', 'speaker': 'User', 'text': 'residence=Shanghai'}]}]}
    qs = [{'sample_id': 'one-session', 'qa_id': 'q', 'question': 'current residence?', 'answer': 'Shanghai', 'evidence': ['last']}]
    result = collect(episode, qs, runtime, 0)
    assert result['local_B']
    assert all(item['position'] == [0, 1] for group in result['local_B'] for item in group)
    assert result['skipped_local_anchors'] == 1


def test_delayed_local_does_not_show_future_to_candidate(setup):
    cfg, data = setup
    cfg['joint']['local_anchors'] = 10
    runtime, _, _, _ = make_runtime(cfg, False)
    episodes, qs = corpus(data / 'train')
    terminal = dict(qs[-1])
    terminal.pop('eval_session')
    result = collect(episodes[0], [terminal], runtime, 0)
    early = next(group for group in result['local_B'] if group[0]['position'][0] == 0)
    for candidate in early:
        assert candidate['evaluation_horizon'] == 'terminal_continuation'
        assert all('Shanghai' not in json.dumps(row['messages']) for row in candidate['rows'])
        assert len(candidate['evaluations']) == 1
    continuation = [row for row in runtime.calls if '/continuation-' in row['event']]
    assert continuation and all(row['evaluator_only'] for row in continuation)


def test_retrieval_stops_before_oversized_model_call(setup):
    cfg, _ = setup
    runtime, _, _, _ = make_runtime(cfg, False)
    state = State('budget', runtime.ranker)
    turn = [{'turn_id': 't', 'speaker': 'User', 'text': 'residence=' + 'x' * 1500}]
    state.observe({'session_id': 's', 'time': '2026-01-01'}, turn, (0, 0), runtime.count)
    state, _, _ = builder(state, turn, complete(tool('update_summary', content='residence=' + 'x' * 1500)), 'b')
    q = {'question': 'current residence?', 'answer': 'x', 'qa_id': 'q'}
    root = prompt(q, runtime.mode)
    cfg['retrieval']['max_prompt_tokens'] = runtime.count(json.dumps(root, ensure_ascii=False)) + 20
    cfg['retrieval']['max_observation_tokens'] = 10000
    with pytest.raises(ValueError, match='context budget'):
        sample(state, q, runtime, 'budget', forest=False, greedy=True)
    assert len(runtime.calls) == 1
