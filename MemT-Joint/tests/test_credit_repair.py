import copy
from types import SimpleNamespace

import pytest

from experiments.io import write_json, write_jsonl, sha256
from experiments.window import reuse_baseline
from methods.joint.credit import construction_reward, score_round


def test_mixed_updater_failures_do_not_saturate_global_and_local(monkeypatch):
    import methods.joint.credit as credit
    monkeypatch.setattr(credit, 'score_record', lambda record, runtime: record['quality'])
    runtime = SimpleNamespace(config={'window': {'blocks': 8},
        'reward': {'invalid': -3, 'memory_cost': 0, 'edit_cost': 0},
        'experiment': {'train_roles': ['U']},
        'joint': {'branch_weights': {'U': {'global': .5, 'local': .5}}}})
    candidates = [[{'role': 'U', 'position': [0, 0], 'valid': v} for v in values]
                  for values in ([True, False], [True, True, False])]
    paths = [{'rows': rows, 'evaluations': [{'quality': .6, 'eval_position': [0, 0]}],
              'snapshots': [{'cost': 0, 'position': [0, 0]}], 'edits': {(0, 0): 0}}
             for rows in candidates]
    local = [dict(rows=copy.deepcopy(rows), evaluations=[{'quality': .6, 'question': {}}], before=[{'quality': 0, 'question': {}}],
                  memory_cost=0, before_cost=0, edit_cost=0, position=[0, 0], anchor_hash='same')
             for rows in candidates]
    data = dict(episode='e', iteration=0, local_B=[], local_U=[local], local_R=[], **{'global': paths})
    rows, groups = score_round(data, runtime)
    for group in groups.values():
        assert [g['reward'] for g in group] == pytest.approx([-1.2, -.6])
        assert all(r['advantage'] < 0 for r in group[0]['rows'])
        assert all(r['advantage'] > 0 for r in group[1]['rows'])
    assert sum(r['weight'] for r in rows) == pytest.approx(1)
    assert construction_reward([{'valid': True}], .6, runtime) == .6
    assert construction_reward([{'valid': False}], .6, runtime) == -3
    runtime.config.pop('window')
    assert construction_reward(candidates[0], .6, runtime) == -3  # Historical protocol unchanged.


def baseline_fixture(tmp_path):
    source, destination = tmp_path / 'old', tmp_path / 'new'
    destination.mkdir()
    validation = source / 'validation/zero'
    contract = dict(common_protocol={'name': 'memt-window-v2'}, models={'base': 'hash'}, data={'a': 'hash'},
        validation={'q': 'hash'}, window_artifacts={}, environment={'python': 'private'}, python='3.11',
        code={'methods/joint/credit.py': 'old', 'models/joint.py': 'unchanged'},
        config={'runtime': {'stop_file': 'old'}, 'window': {'blocks': 8}})
    write_json(source / 'manifest.json', contract)
    write_json(validation / 'manifest.json', {'training_manifest_sha256': sha256(source / 'manifest.json')})
    write_jsonl(validation / 'answers.jsonl', [{'qa_id': i} for i in range(81)])
    summary = dict(policy_version=0, fixture_only=False, count=81, protocol_contract=contract['common_protocol'])
    write_json(validation / 'summary.json', summary)
    write_json(validation / 'COMPLETE.json', {'answers_sha256': sha256(validation / 'answers.jsonl')})
    write_json(source / 'BASELINE.json', dict(step=0, validation='validation/zero', summary_sha256=sha256(validation / 'summary.json')))
    new = copy.deepcopy(contract)
    new['code']['methods/joint/credit.py'] = 'new'
    new['config']['runtime']['stop_file'] = 'new'
    new['config']['window']['baseline_source'] = str(source)
    return source, destination, new, summary


def test_reuse_baseline_preserves_new_manifest_and_binds_source(tmp_path):
    source, destination, new, summary = baseline_fixture(tmp_path)
    write_json(destination / 'manifest.json', {'new': True})
    assert reuse_baseline(source, new, destination) == summary
    assert (destination / 'manifest.json').read_text().find('new') >= 0
    assert (destination / 'REUSED_BASELINE.json').is_file()


@pytest.mark.parametrize('change', ['model', 'data', 'validation', 'code', 'config', 'answers'])
def test_baseline_rejects_changed_inference_or_unbound_results(tmp_path, change):
    source, destination, new, _ = baseline_fixture(tmp_path)
    if change == 'model':
        new['models']['base'] = 'other'
    elif change == 'data':
        new['data']['a'] = 'other'
    elif change == 'validation':
        new['validation']['q'] = 'other'
    elif change == 'code':
        new['code']['models/joint.py'] = 'other'
    elif change == 'config':
        new['config']['window']['blocks'] = 4
    else:
        write_jsonl(source / 'validation/zero/answers.jsonl', [])
    with pytest.raises(ValueError):
        reuse_baseline(source, new, destination)
