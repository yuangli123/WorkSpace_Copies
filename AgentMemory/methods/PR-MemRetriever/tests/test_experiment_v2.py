import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from experiments.io import load_config, read_json, sha256, write_json, write_jsonl
from experiments.construction import subset, compare


def test_fixed_subset_is_balanced_and_preserves_source_order():
    source = [dict(qa_id=str(i), category=i % 4) for i in range(20)]
    left = subset(source, 8, 42)
    assert left == subset(source, 8, 42)
    assert [sum(x['category'] == c for x in left) for c in range(4)] == [2]*4
    assert [int(x['qa_id']) for x in left] == sorted(int(x['qa_id']) for x in left)


def test_profiles_resolve_roles_batch_and_formal_budget_policy():
    from types import SimpleNamespace
    from cli import get_config
    cfg = load_config('configs/experiments/construction-smoke4.yaml')
    assert cfg['distributed']['gpus'] * cfg['construction']['per_device_batch_size'] * cfg['construction']['gradient_accumulation'] == 32
    assert cfg['construction']['steps'] == 4 and cfg['construction']['checkpoint_policy'] == 'export_only'
    assert cfg['construction']['protocol'] == 'paper-v2'
    overridden = get_config(SimpleNamespace(config='configs/experiments/construction-smoke4.yaml', set=['construction.learning_rate=5e-06']))
    assert isinstance(overridden['construction']['learning_rate'], float)
    assert read_json(cfg['construction']['deepspeed'])['optimizer']['type'] == 'AdamW'
    full = load_config('configs/experiments/construction-200.yaml')
    assert full['construction']['steps'] == 200 and full['construction']['checkpoint_policy'] == 'full'
    assert full['experiment']['questions']['evaluation'] == 1307
    for name in ('baseline', 'retrieval-only', 'construction-only', 'construction-and-retrieval'):
        case = load_config('configs/experiments/' + name + '.yaml')
        assert len(case['models']) == 5


def test_retrieval_reward_matches_frozen_pre_migration_implementation():
    from methods.retrieval import tree as new
    cfg = load_config('configs/memt4b_4gpu_100gb.yaml')
    def forest(module):
        return [[module.Node(0, -1, 0, []), module.Node(1, 0, 1, [], children=[2,3], retrieved=[{'source_turn_ids':['D1:1']}]),
                 module.Node(2, 1, 2, [], terminal=True, answer='Paris', retrieved=[{'source_turn_ids':['D1:1']}]),
                 module.Node(3, 1, 2, [], terminal=True, answer='Rome')],
                [module.Node(0, -1, 0, []), module.Node(1, 0, 1, [], terminal=True, answer='Paris')]]
    qa = {'answer':'Paris','evidence':['D1:1']}
    left = read_json('tests/fixtures/retrieval_reward_legacy.json')['nodes']
    right = new.score_forest(forest(new), qa, cfg)
    assert left == [[asdict(n) for n in t] for t in right]


def comparison_fixture(tmp_path):
    cfg = load_config('configs/experiments/construction-smoke4.yaml')
    cfg['experiment']['questions']['evaluation'] = 2
    for variant in ('base','trained'):
        questions = [dict(sample_id='s',qa_id=str(i),category=1,question='Where '+str(i)+'?',answer='Paris') for i in range(2)]
        answers, judgments = [], []
        for i, qa in enumerate(questions):
            pred = 'Paris' if variant == 'trained' or i == 0 else 'Rome'
            row = dict(qa,gold='Paris',pred=pred,response=pred,f1=float(pred=='Paris'),bleu1=float(pred=='Paris'),finished=True)
            answers.append(row)
            label = 'CORRECT' if pred=='Paris' else 'WRONG'
            judgments.append({**{k:row[k] for k in ('sample_id','qa_id','question','gold','pred','category')},
                              'judge_label':label,'judge_reasoning':'same topic', 'finish_reason':'stop',
                              'judge_raw_output':json.dumps({'label':label,'reasoning':'same topic'})})
        evaluation, judge = tmp_path/('evaluate-'+variant), tmp_path/('judge-'+variant)
        write_jsonl(evaluation/'answers.jsonl.gz', answers)
        write_jsonl(tmp_path/('memory-eval-'+variant)/'questions.jsonl', questions)
        effective = copy.deepcopy(cfg)
        effective['models']['construction'] = variant
        effective['model']['checkpoint'] = str(Path(cfg['models']['retrieval']).resolve())
        write_json(evaluation/'manifest.json', {'config':effective,'code_sha256':{},'packages':{}})
        write_jsonl(judge/'judgments.jsonl', judgments)
        count = sum(x['judge_label']=='CORRECT' for x in judgments)
        answer_sha = sha256(evaluation/'answers.jsonl.gz')
        write_json(judge/'manifest.json', {'spec':{'answers_sha256':answer_sha,'same_protocol':True}})
        write_json(judge/'summary.json', {'count':2,'correct':count,'wrong':2-count,'llm_as_judge':count/2,
                   'error':0,'skipped':0,'answers_sha256':answer_sha,'judge_results_sha256':sha256(judge/'judgments.jsonl')})
        write_json(judge/'COMPLETE.json', {'summary_sha256':sha256(judge/'summary.json'),'judgments_sha256':sha256(judge/'judgments.jsonl')})
    return cfg


def test_comparison_recomputes_metrics_and_rejects_protocol_drift(tmp_path):
    cfg = comparison_fixture(tmp_path)
    result = compare(tmp_path,cfg)
    assert result['delta_percentage_points']['f1'] == 50
    assert result['delta_percentage_points']['llm_as_judge'] == 50
    path = tmp_path/'evaluate-trained/manifest.json'
    value = read_json(path)
    value['config']['evaluation']['max_steps'] += 1
    write_json(path,value)
    with pytest.raises(ValueError,match='settings/source'):
        compare(tmp_path,cfg)


def test_stage_reuse_checks_inputs_and_output_hashes(tmp_path, monkeypatch):
    pytest.importorskip('fcntl')
    from experiments.stages import Stages
    model = tmp_path/'model'
    write_json(model/'config.json',{})
    write_json(model/'tokenizer.json',{})
    (model/'model.safetensors').write_bytes(b'identity fixture only')
    cfg={'models':{k:str(model) for k in ('base','construction','retrieval','embedding','judge')},
         'storage':{'allowed_root':str(tmp_path),'roots':[str(tmp_path)],'ledger':str(tmp_path/'ledger.json'),
                    'limit_bytes':10000000,'warning_bytes':9000000,'runtime_reserve_bytes':1024}}
    stages=Stages(cfg,tmp_path,tmp_path/'run')
    monkeypatch.setattr(stages, 'check_budget', lambda reason: stages.budget.check(0, reason))
    folder=tmp_path/'run/output'
    command=[sys.executable,'-c',f'from pathlib import Path; p=Path({str(folder)!r}); p.mkdir(); (p/"value").write_text("ok")']
    stages.run('one',command,folder)
    stages.run('one',command,folder)  # would fail mkdir if the stage actually ran twice
    (folder/'value').write_text('changed')
    with pytest.raises(ValueError,match='output changed'):
        stages.run('one',command,folder)
