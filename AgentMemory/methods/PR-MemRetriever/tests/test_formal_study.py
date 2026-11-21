import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.io import load_config, read_json, read_jsonl, write_json, write_jsonl


def test_memory_r2_prompt_preserves_user_rule_and_accepts_label_only():
    from evaluation.prompts import judge_prompts, LOCOMO_SYSTEM_PROMPT
    from evaluation.judge import verdict
    system, template = judge_prompts('memory-r2-generous-v1')
    assert 'same topic/date' in template and 'Be generous' in template
    assert 'Question: Q Gold answer: G Generated answer: A' in template.format(question='Q', gold_answer='G', generated_answer='A')
    assert verdict('{"label":"WRONG"}', require_reasoning=False) == {'label': 'WRONG'}
    with pytest.raises(AssertionError):
        verdict('{"label":"WRONG"}')
    assert judge_prompts('legacy-topic-v1')[0] == LOCOMO_SYSTEM_PROMPT


def test_formal_profile_has_genuine_rollouts_export_only_and_same_judge():
    cfg = load_config('configs/experiments/construction-study-200.yaml')
    assert cfg['construction']['steps'] == 200
    assert cfg['construction']['checkpoint_policy'] == 'export_only'
    assert cfg['construction']['minimum_valid_operations'] >= 20000
    assert cfg['construction']['rollouts'] > 42
    assert cfg['construction']['per_device_batch_size'] * cfg['construction']['gradient_accumulation'] * 4 == 32
    assert cfg['experiment']['questions'] == {'train': 152, 'evaluation': 1307}
    assert cfg['study']['judge_protocol'] == 'memory-r2-generous-v1'
    assert cfg['runtime']['compress_forests'] and cfg['runtime']['resumable_evaluation']


def test_new_chroma_collections_use_explicit_thread_limit():
    from memory.vector_db import ChromaVectorDB
    db = ChromaVectorDB.__new__(ChromaVectorDB)
    db.config = SimpleNamespace(hnsw_num_threads=1)
    db.default_embedding_function = 'fixture'
    calls = []
    db.client = SimpleNamespace(create_collection=lambda **kwargs: calls.append(kwargs))
    db.create_collection('new')
    assert calls[-1]['configuration'] == {'hnsw': {'num_threads': 1}}
    db.create_collection('explicit', configuration={'hnsw': {'num_threads': 2}})
    assert calls[-1]['configuration']['hnsw']['num_threads'] == 2


def test_completed_rollouts_resume_without_reembedding(tmp_path, monkeypatch):
    from experiments import workflow, progress
    from test_construction_v2 import RawPolicy, episode
    cfg = load_config('configs/memt4b.yaml')
    cfg['memory'].update(backend='lexical_test', summary_context_turns=1)
    cfg['construction'].update(protocol='paper-v2', rollouts=4, minimum_valid_operations=3)
    data, output = tmp_path / 'data', tmp_path / 'memory'
    write_jsonl(data / 'episodes.jsonl', [episode()])
    write_jsonl(data / 'questions.jsonl', [dict(qa_id='q', sample_id='s', question='Where?', answer='Rome', category=1, evidence=[])])
    write_json(data / 'partition.json', {'split': 'train'})
    policy = RawPolicy()
    workflow.build(cfg, data, output, policy)
    assert policy.calls == 4  # two independently seeded episodes, two chunks each
    before = read_jsonl(output / 'operations.jsonl')
    assert len({op['op_id'] for op in before}) == 4
    assert len({op['sampling_seed'] for op in before}) == 4
    def fail(*args):
        raise AssertionError('Completed episodes must not restore or re-embed')
    monkeypatch.setattr(progress, 'restore_chunk', fail)
    second = RawPolicy()
    workflow.build(cfg, data, output, second, resume=True)
    assert second.calls == 0 and read_jsonl(output / 'operations.jsonl') == before
    assert read_json(output / 'COMPLETE.json')['eligible_operations'] == 4


def test_streaming_hindsight_means_leaves_then_questions():
    from methods.construction.credit import hindsight_scores
    operations = [dict(op_id='a', sample_id='s', source_turn_ids=['t']),
                  dict(op_id='b', sample_id='s', source_turn_ids=[])]
    questions = [dict(qa_id='1', sample_id='s', evidence=['t']), dict(qa_id='2', sample_id='s', evidence=[])]
    def node(adv, ops):
        return dict(children=[], advantage=adv, retrieved=[dict(operation_ids=ops)])
    records = [dict(qa_id='1', sample_id='s', trees=[[{}, node(2, ['b']), node(-1, [])]]),
               dict(qa_id='2', sample_id='s', trees=[[{}, node(3, ['a', 'b'])]])]
    cfg = {'construction': {'evidence_gate': True, 'trace_gate': True, 'trace_lambda': .1}}
    left = hindsight_scores(operations, iter(records), questions, cfg)
    assert left[0]['score'] == pytest.approx(.4)  # (mean(2,-1) + .3) / 2
    assert left[1]['score'] == pytest.approx(.2)  # (mean(.2,0) + .3) / 2
    assert left[0]['evidence_hits'] == 2 and left[1]['trace_hits'] == 2
    with pytest.raises(ValueError, match='Duplicate'):
        hindsight_scores(operations, iter(records + records[:1]), questions, cfg)


def test_compressed_forests_resume_and_do_not_repeat_sampling(tmp_path, monkeypatch):
    from experiments import workflow
    cfg = load_config('configs/memt4b.yaml')
    cfg['runtime'] = {'compress_forests': True}
    cfg['construction']['forest_batch_size'] = 1
    questions = [dict(sample_id='s', qa_id=str(i)) for i in range(2)]
    write_json(tmp_path / 'memory/index.json', {})
    write_jsonl(tmp_path / 'memory/questions.jsonl', questions)
    memory = SimpleNamespace(path=tmp_path / 'memory', questions=questions, environment=lambda *args: None)
    calls = []
    def sample(policy, envs, batch, config, seeds):
        calls.extend(seeds)
        return [[] for _ in batch]
    monkeypatch.setattr(workflow, 'sample_forests', sample)
    monkeypatch.setattr(workflow, 'forest_record', lambda q, f, step, seed: dict(q, trees=f, step=step, seed=seed))
    workflow.collect_forests(cfg, memory, None, tmp_path / 'forests')
    workflow.collect_forests(cfg, memory, None, tmp_path / 'forests', resume=True)
    assert calls == [cfg['seed'], cfg['seed'] + 1]
    assert [r['qa_id'] for r in read_jsonl(tmp_path / 'forests/forests.jsonl.gz')] == ['0', '1']
    value = read_json(tmp_path / 'forests/forests/000000.json.gz')
    value['seed'] += 1
    write_json(tmp_path / 'forests/forests/000000.json.gz', value)
    with pytest.raises(ValueError, match='sampling seed'):
        workflow.collect_forests(cfg, memory, None, tmp_path / 'forests', resume=True)


def test_evaluation_resumes_sealed_rows_and_checks_identity(tmp_path, monkeypatch):
    from evaluation import retrieval
    from methods.retrieval.tree import Node
    cfg = load_config('configs/memt4b.yaml')
    cfg['runtime'] = {'resumable_evaluation': True}
    questions = [dict(sample_id='s', qa_id=str(i), question='Where?', answer='Paris', category=1) for i in range(2)]
    memory = SimpleNamespace(questions=questions, environment=lambda *args: None)
    calls = []
    def sample(policy, env, qa, config, rng):
        calls.append(qa['qa_id'])
        if qa['qa_id'] == '1' and len(calls) == 2:
            raise RuntimeError('test interruption')
        return [Node(0, -1, 0, []), Node(1, 0, 1, [], valid=True, terminal=True, answer='Paris')]
    monkeypatch.setattr(retrieval, 'sample_tree', sample)
    target = tmp_path / 'answers.jsonl.gz'
    with pytest.raises(RuntimeError, match='interruption'):
        retrieval.evaluate(cfg, memory, None, target)
    retrieval.evaluate(cfg, memory, None, target)
    assert calls == ['0', '1', '1']
    assert len(read_jsonl(target)) == 2
    (tmp_path / 'questions/000000.json.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='seal'):
        retrieval.evaluate(cfg, memory, None, target)


def test_stage_defers_before_starting_sft_without_enough_time(tmp_path, monkeypatch):
    pytest.importorskip('fcntl')
    from experiments import stages as module
    model = tmp_path / 'model'
    write_json(model / 'config.json', {})
    write_json(model / 'tokenizer.json', {})
    (model / 'model.safetensors').write_bytes(b'fixture')
    cfg = {'models': {k: str(model) for k in ('base', 'construction', 'retrieval', 'embedding', 'judge')},
           'storage': {'allowed_root': str(tmp_path), 'roots': [str(tmp_path)], 'ledger': str(tmp_path / 'ledger.json'),
                       'limit_bytes': 10000000, 'warning_bytes': 9000000, 'runtime_reserve_bytes': 1},
           'study': {'walltime_margin_seconds': 600}}
    stages = module.Stages(cfg, tmp_path, tmp_path / 'run')
    monkeypatch.setattr(module, 'remaining_seconds', lambda: 1000)
    with pytest.raises(module.StageDeferred):
        stages.run('sft', ['must-not-start'], tmp_path / 'run/sft', minimum_seconds=14400)
    assert not (tmp_path / 'run/stages/sft.json').exists()


def test_dispatch_records_one_job_and_does_not_duplicate_live_work(tmp_path, monkeypatch):
    pytest.importorskip('fcntl')
    from experiments import study_dispatch as dispatch
    monkeypatch.setattr(dispatch, '__file__', str(tmp_path / 'experiments/study_dispatch.py'))
    monkeypatch.setattr(dispatch.os, 'getuid', lambda: 2086)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / 'config.json'
    run = tmp_path / 'artifacts/study'
    write_json(config, {'experiment': {'output': str(run)}, 'study': {'maximum_jobs': 2}})
    calls, live = [], []
    def check(command, **kwargs):
        calls.append(command)
        if command[0] == 'squeue':
            return '\n'.join(live)
        job = str(101 + len(live))
        live.append(job)
        return job + '\n'
    monkeypatch.setattr(dispatch.subprocess, 'check_output', check)
    dispatch.main(['--config', str(config)])
    dispatch.main(['--config', str(config)])
    assert sum(c[0] == 'sbatch' for c in calls) == 1
    assert read_json(run / 'SUBMITTED.json')['job'] == '101'
    write_json(run / 'DEFERRED.json', {'job': '101'})
    monkeypatch.setenv('SLURM_JOB_ID', '101')
    dispatch.main(['--config', str(config), '--after-job', '101'])
    assert sum(c[0] == 'sbatch' for c in calls) == 2
    assert '--dependency=afterok:101' in calls[-1]


def test_new_pair_comparison_validates_complete_judge_denominator(tmp_path):
    from experiments.construction_study import compare
    cfg = load_config('configs/experiments/construction-study-200.yaml')
    cfg['experiment']['questions']['evaluation'] = 1
    qa = dict(sample_id='s', qa_id='q', question='Where?', answer='Paris', category=1)
    write_jsonl(tmp_path / 'memory-test/questions.jsonl', [qa])
    write_json(tmp_path / 'memory-test/index.json', {})
    from experiments.io import sha256
    for name, model in [('construction_only', cfg['models']['base']), ('construction_retrieval', cfg['models']['retrieval'])]:
        evaluation, judge = tmp_path / ('evaluate-' + name), tmp_path / ('judge-' + name)
        row = dict(qa, gold='Paris', pred='Paris', response='Paris', f1=1., bleu1=1., finished=True)
        write_jsonl(evaluation / 'answers.jsonl.gz', [row])
        recorded = copy.deepcopy(cfg)
        recorded['model']['checkpoint'] = str(Path(model).resolve())
        write_json(evaluation / 'manifest.json', {'config': recorded, 'inputs': {}, 'code_sha256': {}, 'packages': {}})
        judgment = dict(row, judge_label='CORRECT', judge_raw_output='{"label":"CORRECT"}', finish_reason='stop')
        write_jsonl(judge / 'judgments.jsonl', [judgment])
        answer_sha = sha256(evaluation / 'answers.jsonl.gz')
        write_json(judge / 'manifest.json', {'spec': {'protocol': cfg['study']['judge_protocol'], 'answers_sha256': answer_sha}})
        write_json(judge / 'summary.json', {'count': 1, 'correct': 1, 'wrong': 0, 'error': 0, 'skipped': 0, 'llm_as_judge': 1.,
                   'answers_sha256': answer_sha, 'judge_results_sha256': sha256(judge / 'judgments.jsonl')})
        write_json(judge / 'COMPLETE.json', {'summary_sha256': sha256(judge / 'summary.json'),
                                          'judgments_sha256': sha256(judge / 'judgments.jsonl')})
    assert compare(tmp_path, cfg)['variants']['construction_only']['f1'] == 1
    path = tmp_path / 'judge-construction_only/summary.json'
    value = read_json(path)
    value['count'] = 2
    write_json(path, value)
    with pytest.raises(ValueError, match='coverage/hash'):
        compare(tmp_path, cfg)
