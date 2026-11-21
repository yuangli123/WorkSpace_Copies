import re
from pathlib import Path

import pytest

from experiments.joint import make_runtime, metrics
from experiments.joint_config import load, validate
from experiments.io import read_jsonl, write_json
from methods.joint.rollout import blocks, new_state, formation, evolution, continue_state
from methods.joint.batching import formation_many, evolution_many, continuation_many
from methods.joint.short_episode import collect_short
from methods.joint.credit import score_round
from scripts.prepare_short_episodes import prepare

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    config = load(ROOT/'configs/fixture.json')
    config['short_episode'] = {'blocks': 8}
    config['memory']['summary_context_turns'] = 4
    config['joint'].update(global_questions=4, local_questions=4, retrieval_questions=8)
    episode = dict(sample_id='short-fixture', sessions=[dict(session_id='s', time='2026-01-01', turns=[
        dict(turn_id='t%d' % i, speaker='User', text='residence=City%d' % i) for i in range(12)])])
    questions = [dict(sample_id=episode['sample_id'], qa_id='q%d' % i,
        question='What is my residence number %d?' % i, answer='GOLD_SECRET_%d' % i,
        evidence=['t%d' % i], category=1, reviewed=True, eval_session=0) for i in range(8)]
    return config, episode, questions


def test_short_preserves_branches_mass_and_keeps_gold_out_of_policy():
    config, episode, qs = fixture()
    config['joint'].update(global_rollouts=4, builder_rollouts=4, updater_rollouts=4)
    runtime, policy, _, _ = make_runtime(config, False)
    record = collect_short(episode, qs, runtime, 0)
    rows, _ = score_round(record, runtime)
    info = metrics(record, runtime, rows)
    assert len(record['global']) == 4
    assert len(record['local_B'][0]) == len(record['local_U'][0]) == 4
    assert len(record['local_R']) == len(qs)
    assert info['role_weights'] == pytest.approx(dict(B=1/3, U=1/3, R=1/3))
    assert {r['branch'] for r in rows if r['role'] == 'B'} == {'global', 'local'}
    assert {r['branch'] for r in rows if r['role'] == 'U'} == {'global', 'local'}
    first = [r for r in runtime.calls if '/global-' in r['event'] and '/block-0/B' in r['event']]
    empty = new_state(episode['sample_id'], runtime)
    block = next(blocks(episode, config['memory']['summary_context_turns']))
    empty.observe(block.session, block.turns, block.position, runtime.count)
    assert len(first) == 4 and all(r['state_hash'] == empty.fingerprint() for r in first)
    assert not any('GOLD_SECRET' in str(messages) for messages in policy.calls)
    assert record['short_episode']['remaining_blocks'] <= 7


def test_batch_matches_serial_states_and_per_branch_update_order():
    config, episode, _ = fixture()
    runtime, _, _, _ = make_runtime(config, False)
    sequence = list(blocks(episode, config['memory']['summary_context_turns']))
    states = [new_state(episode['sample_id'], runtime) for _ in range(3)]
    for state in states:
        state.observe(sequence[0].session, sequence[0].turns, sequence[0].position, runtime.count)
    originals = [s.fingerprint() for s in states]
    events = ['branch-%d' % i for i in range(3)]
    serial_b = [formation(s, sequence[0], runtime, e+'/B', greedy=True) for s, e in zip(states, events)]
    serial_u = [evolution(f[0], f[1], sequence[0], runtime, e+'/U', greedy=True) for f, e in zip(serial_b, events)]
    batched_b = formation_many(states, sequence[0], runtime, [e+'/B' for e in events], greedy=True)
    batched_u = evolution_many([f[0] for f in batched_b], [f[1] for f in batched_b], sequence[0], runtime,
                               [e+'/U' for e in events], greedy=True)
    assert [s.fingerprint() for s in states] == originals
    assert [u[0].fingerprint() for u in serial_u] == [u[0].fingerprint() for u in batched_u]
    # Use identical identities to compare future greedy B/U rollouts as well.
    serial = [continue_state(u[0], sequence[1:], runtime, e) for u, e in zip(serial_u, events)]
    batched = continuation_many([u[0] for u in batched_u], sequence[1:], runtime, events)
    assert [s.fingerprint() for s in serial] == [s.fingerprint() for s in batched]


def test_reject_unreviewed_or_future_qa_and_excess_length():
    config, episode, qs = fixture()
    runtime, _, _, _ = make_runtime(config, False)
    qs[0]['eval_session'] = 1
    with pytest.raises(ValueError, match='visible endpoint'):
        collect_short(episode, qs, runtime, 0)
    qs[0]['eval_session'] = 0
    qs[0]['reviewed'] = False
    with pytest.raises(ValueError, match='reviewed'):
        collect_short(episode, qs, runtime, 0)
    episode['sessions'][0]['turns'] *= 3
    with pytest.raises(ValueError, match='2..8'):
        collect_short(episode, qs, runtime, 0)


def test_reviewed_train_data_and_audit_tampering(tmp_path):
    source = ROOT/'artifacts/short8-r2-20261113'
    if not (source/'TRAIN_BLOCKS.json').exists():
        pytest.skip('Local TRAIN-only transcript is not bundled with source')
    output = tmp_path/'train'
    assert prepare(source/'TRAIN_BLOCKS.json', source/'reviewed_qa.txt', output) == (14,112)
    episodes, qs = read_jsonl(output/'episodes.jsonl'), read_jsonl(output/'questions.jsonl')
    assert all(e['source_sample_id'] == 'conv-26' for e in episodes)
    assert all(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]+[a-zA-Z0-9]',e['sample_id']+'_facts') for e in episodes)
    assert all(len(list(blocks(e, 4))) <= 8 for e in episodes)
    for e in episodes:
        visible = {t['turn_id'] for s in e['sessions'] for t in s['turns']}
        for q in [q for q in qs if q['sample_id'] == e['sample_id']]:
            assert set(q['evidence']) <= visible and q['eval_session'] == len(e['sessions'])-1
    from experiments.joint import identity
    config, _, _ = fixture()
    assert identity(config, output)['data']['short_episode_audit']
    (output/'questions.jsonl').write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='changed after review'):
        identity(config, output)


def test_production_clip_ids_work_in_real_chroma(tmp_path):
    chroma = pytest.importorskip('chromadb')
    from chromadb.config import Settings
    client = chroma.PersistentClient(path=str(tmp_path/'chroma'), settings=Settings(anonymized_telemetry=False))
    with pytest.raises(Exception):
        client.create_collection('conv-26/short-00_facts', embedding_function=None)
    for i in range(14):
        for kind in ('summary','facts','experiences','personas','turns'):
            name = 'conv-26-short-%02d_%s' % (i,kind)
            assert client.create_collection(name, embedding_function=None).name == name


def test_full_validation_contract_unchanged_by_short_sampling():
    from experiments.joint_config import contract
    old, new = [load(ROOT/'configs'/p) for p in ('memt-execution-m4-50.json', 'short8-m4-50.json')]
    validate(new, training=True)
    assert contract(old) == contract(new)
    assert old['joint'] == new['joint']
    assert new['retrieval']['steps'] == 50
    assert new['runtime']['validate_every'] == 10


@pytest.mark.parametrize('all_global_paths_empty', [False, True])
def test_local_u_uses_other_global_paths_before_bounded_builder_resample(monkeypatch, all_global_paths_empty):
    import methods.joint.short_episode as short
    config, episode, qs = fixture()
    runtime, _, _, _ = make_runtime(config, False)
    original = short.formation_many
    hashes, repair_calls = {}, []

    def controlled(states, block, runtime, events, greedy=False):
        # Only block 1 has both new-write and retention QA in this fixture.
        results = original(states, block, runtime, events, greedy=True)
        if '/anchor-repair/' in events[0]:
            repair_calls.extend(events)
        if '/global-' in events[0] and block.turns[0]['turn_id'] == 't4':
            for gi, (state, output) in enumerate(zip(states, results)):
                if all_global_paths_empty or gi == 0:
                    output[2][0]['valid'] = False
                    results[gi] = (state.clone(), [], output[2], [])
                else:
                    hashes[gi] = output[0].fingerprint()
        return results

    monkeypatch.setattr(short, 'formation_many', controlled)
    record = short.collect_short(episode, qs, runtime, 7)
    info = record['short_episode']
    assert info['anchor_block'] == 1
    assert info['anchor_builder_resampled'] == all_global_paths_empty
    assert len(repair_calls) == (config['joint']['builder_rollouts'] if all_global_paths_empty else 0)
    if not all_global_paths_empty:
        assert info['anchor_global_path'] == 1
        assert {c['anchor_hash'] for c in record['local_U'][0]} == {hashes[1]}
    rows, _ = score_round(record, runtime)
    measured = metrics(record, runtime, rows)
    assert measured['role_weights'] == pytest.approx(dict(B=1/3, U=1/3, R=1/3))
    assert {r['branch'] for r in rows if r['role'] == 'U'} == {'global', 'local'}
    selected = [r['question'] for r in record['local_U'][0][0]['evaluations']]
    assert len(selected) == len({q['qa_id'] for q in selected}) == config['joint']['local_questions']
    assert any(set(q['evidence']) & {'t4', 't5', 't6', 't7'} for q in selected)
    assert any(set(q['evidence']) <= {'t0', 't1', 't2', 't3'} for q in selected)


def test_local_u_skips_missing_qa_but_keeps_local_b_and_global_training():
    config, episode, qs = fixture()
    for q in qs:
        q['evidence'] = ['t0']
    runtime, _, _, _ = make_runtime(config, False)
    record = collect_short(episode, qs, runtime, 0)
    assert record['short_episode']['skipped_branches'] == {'U.local': 'no_qa_coverage'}
    assert record['local_U'] == [] and len(record['local_B'][0]) == config['joint']['builder_rollouts']
    assert record['short_episode']['anchor_block'] == 2  # B-only fallback uses the terminal block.
    assert not any('/anchor-repair/' in r['event'] for r in runtime.calls)
    rows, _ = score_round(record, runtime)
    info = metrics(record, runtime, rows)
    from methods.joint.short_episode import validate_short_weights
    validate_short_weights(info, config['joint']['branch_weights'], info['skipped_branches'])
    assert info['role_weights'] == pytest.approx(dict(B=1/3, U=1/6, R=1/3))
    assert info['active_weight_mass'] == pytest.approx(5/6)


def test_local_u_bounds_builder_repair_and_never_invents_pending_items(monkeypatch):
    import methods.joint.short_episode as short
    config, episode, qs = fixture()
    runtime, _, _, _ = make_runtime(config, False)
    original = short.formation_many
    repairs = []

    def empty_candidates(states, block, runtime, events, greedy=False):
        results = original(states, block, runtime, events, greedy)
        repair = '/anchor-repair/' in events[0]
        if repair:
            repairs.extend(events)
        if repair or '/global-' in events[0] and block.turns[0]['turn_id'] == 't4':
            return [(s.clone(), [], f[2], []) for s, f in zip(states, results)]
        return results

    monkeypatch.setattr(short, 'formation_many', empty_candidates)
    record = short.collect_short(episode, qs, runtime, 0)
    assert len(repairs) == config['joint']['builder_rollouts']
    assert record['local_U'] == []
    assert record['short_episode']['skipped_branches'] == {'U.local': 'no_legal_candidates'}
    rows, _ = score_round(record, runtime)
    info = metrics(record, runtime, rows)
    short.validate_short_weights(info, config['joint']['branch_weights'], info['skipped_branches'])
    assert info['branch_weights']['U']['local'] == 0
    assert info['branch_weights']['U']['global'] == pytest.approx(1/6)
    assert info['active_weight_mass'] == pytest.approx(5/6)


def test_no_updater_candidates_can_train_b_and_r_without_redistributing_weight(monkeypatch, tmp_path):
    import methods.joint.short_episode as short
    config, episode, qs = fixture()
    runtime, _, _, _ = make_runtime(config, False)
    original = short.formation_many

    def no_updates(states, block, runtime, events, greedy=False):
        results = original(states, block, runtime, events, greedy)
        if '/global-' in events[0] or '/anchor-repair/' in events[0]:
            return [(s.clone(), [], f[2], []) for s, f in zip(states, results)]
        return results

    monkeypatch.setattr(short, 'formation_many', no_updates)
    record = short.collect_short(episode, qs, runtime, 0)
    assert record['short_episode']['skipped_branches'] == {
        'U.global': 'no_legal_candidates', 'U.local': 'no_legal_candidates'}
    rows, _ = score_round(record, runtime)
    info = metrics(record, runtime, rows)
    short.validate_short_weights(info, config['joint']['branch_weights'], info['skipped_branches'])
    assert info['rows_by_role']['U'] == 0
    assert info['role_weights'] == pytest.approx(dict(B=1/3, U=0, R=1/3))
    assert info['active_weight_mass'] == pytest.approx(2/3)
    # Missing U without the explicit skip record remains an error.
    with pytest.raises(ValueError):
        short.validate_short_weights(info, config['joint']['branch_weights'], {})
    with pytest.raises(ValueError, match='Unsupported'):
        short.validate_short_weights(info, config['joint']['branch_weights'], {'B.local': 'no_legal_candidates'})
    # Exercise the runner guard and the persisted metrics, not just the collector.
    from experiments.joint import run
    from experiments.io import read_json, write_jsonl, sha256
    data = tmp_path/'data'
    write_jsonl(data/'episodes.jsonl', [episode])
    write_jsonl(data/'questions.jsonl', qs)
    write_json(data/'partition.json', {'split':'train'})
    write_json(data/'DATA_AUDIT.json', {'version':'memt-short-empty-v1', 'review':'assistant_source_review',
        'scope':'fixture_only', 'files':{n:sha256(data/n) for n in ('episodes.jsonl','questions.jsonl','partition.json')}})
    output = tmp_path/'smoke'
    run(config, data, output, smoke=True)
    assert read_json(output/'SMOKE_COMPLETE.json')['optimizer_updates'] == 0
    saved = read_json(next(output.glob('attempts/*/step-*/metrics.json')))
    assert saved['skipped_branches'] == record['short_episode']['skipped_branches']
    assert saved['rows_by_role']['U'] == 0
    assert saved['active_weight_mass'] == pytest.approx(2/3)


@pytest.mark.parametrize('defect', [None, 'slow', 'missing_local', 'stale_answer', 'save_over_budget', 'skip_local_u', 'skip_all_u'])
def test_profile_gate_rejects_incomplete_acceptance(tmp_path, defect):
    from scripts.verify_short_profile import verify
    root = tmp_path/'profile'
    write_json(root/'train/TRAINING_COMPLETE.json', dict(step=2, fixture_only=False))
    write_json(root/'train/LATEST.json', dict(step=2))
    weights = dict(B={'global':1/6,'local':1/6}, U={'global':1/6,'local':1/6}, R={'mot':1/3})
    write_json(root/'train/manifest.json', dict(config={'backend':'verl','joint':{'branch_weights':weights}}, code={}, models={}, data={}, common_protocol={}))
    for i in range(2):
        m = dict(iteration=i, fixture_only=False, answer_policy_version=i,
            wall_seconds=100, ordinary_step_seconds_including_checkpoint=150, nonzero_advantages=1,
            short_episode={'blocks':8,'initial_state':'empty'}, rows_by_role=dict(B=1,U=1,R=1),
            role_weights=dict(B=1/3,U=1/3,R=1/3),
            branch_weights={r:dict(v) for r,v in weights.items()})
        if defect in ('skip_local_u','skip_all_u'):
            m['short_episode']['skipped_branches'] = {'U.local':'no_legal_candidates'}
            m['branch_weights']['U']['local'] = 0
            m['role_weights']['U'] = 1/6
            if defect == 'skip_all_u':
                m['short_episode']['skipped_branches']['U.global'] = 'no_legal_candidates'
                m['branch_weights']['U']['global'] = 0
                m['role_weights']['U'] = 0
                m['rows_by_role']['U'] = 0
        if defect == 'slow': m['wall_seconds'] = 1801
        if defect == 'save_over_budget': m['ordinary_step_seconds_including_checkpoint'] = 1801
        if defect == 'missing_local': m['branch_weights']['U']['local'] = 0
        if defect == 'stale_answer': m['answer_policy_version'] = -1
        write_json(root/('train/attempts/test/step-%06d/metrics.json' % (i+1)), m)
    if defect in ('slow','missing_local','stale_answer'):
        with pytest.raises(ValueError): verify(root, tmp_path/'PASSED.json')
        assert not (tmp_path/'PASSED.json').exists()
    else:
        receipt = verify(root, tmp_path/'PASSED.json')
        assert receipt['calibration_updates'] == 2
        assert receipt['scope'] == 'training_compute_checkpoint_separate'
        assert receipt['full_step_including_checkpoint_passed'] == (defect != 'save_over_budget')
