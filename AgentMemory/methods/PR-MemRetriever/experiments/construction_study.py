"""Formal construction SFT followed by paired C200+R0 and C200+R200 tests."""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from .construction import data_view, ensure_rows
from .io import load_config, read_json, read_jsonl, sha256, write_json, write_jsonl
from .stages import Stages, StageDeferred, model_identity


def compare(output, config):
    from methods.retrieval.tree import answer_f1, answer_bleu1
    from evaluation.judge import verdict, verify_prefix
    variants, paired, aligned, judge_spec, evaluation_spec = {}, {}, None, None, None
    questions = read_jsonl(output / 'memory-test/questions.jsonl')
    if len(questions) != config['experiment']['questions']['evaluation']:
        raise ValueError('Test question count changed')
    for name, retriever in [('construction_only', config['models']['base']),
                             ('construction_retrieval', config['models']['retrieval'])]:
        folder, judged = output / ('evaluate-' + name), output / ('judge-' + name)
        answers = read_jsonl(folder / 'answers.jsonl.gz')
        judgments = read_jsonl(judged / 'judgments.jsonl')
        if len(answers) != len(questions) or len(judgments) != len(answers):
            raise ValueError('Incomplete paired evaluation')
        verify_prefix(judgments, answers)
        keys = [(a['sample_id'], a['qa_id']) for a in answers]
        if len(set(keys)) != len(keys) or aligned is not None and keys != aligned:
            raise ValueError('Test questions/order differ across retrievers')
        aligned = keys
        recorded = read_json(folder / 'manifest.json')
        normalized = copy.deepcopy(recorded['config'])
        if normalized['model'].pop('checkpoint') != str(Path(retriever).resolve()):
            raise ValueError('Wrong retrieval model in evaluation')
        normalized['storage'].pop('ledger')
        contract = {'config': normalized, 'inputs': recorded['inputs'],
                    'code': recorded['code_sha256'], 'packages': recorded['packages']}
        if evaluation_spec is not None and contract != evaluation_spec:
            raise ValueError('Paired evaluation source, memory, or settings changed')
        evaluation_spec = contract
        spec = read_json(judged / 'manifest.json')['spec']
        if spec['protocol'] != config['study']['judge_protocol']:
            raise ValueError('Wrong judge protocol')
        spec.pop('answers_sha256')
        if judge_spec is not None and spec != judge_spec:
            raise ValueError('Paired judge settings changed')
        judge_spec = spec
        for row, qa, judge in zip(answers, questions, judgments):
            if any(row[k] != qa[k] for k in ('sample_id', 'qa_id', 'question', 'answer', 'category')):
                raise ValueError('Answer not aligned with test input')
            if row['gold'] != qa['answer'] or row['response'] != row['pred']:
                raise ValueError('Gold/prediction aliases differ')
            value = verdict(judge['judge_raw_output'], require_reasoning=False)
            if value['label'] != judge['judge_label'] or judge['finish_reason'] == 'length':
                raise ValueError('Invalid final judge verdict')
            for metric, fn in [('f1', answer_f1), ('bleu1', answer_bleu1)]:
                if not math.isclose(row[metric], fn(row['pred'], row['gold']), abs_tol=1e-12):
                    raise ValueError('Answer score does not recompute')
            key = (row['sample_id'], row['qa_id'])
            paired.setdefault(key, dict(qa)).update({name: {k: row[k] for k in ('pred', 'f1', 'bleu1', 'finished')}})
            paired[key][name]['judge_label'] = judge['judge_label']
        summary, done = read_json(judged / 'summary.json'), read_json(judged / 'COMPLETE.json')
        correct = sum(j['judge_label'] == 'CORRECT' for j in judgments)
        if (summary['count'] != len(answers) or summary['error'] or summary['skipped']
                or summary['correct'] != correct or summary['wrong'] != len(answers) - correct
                or not math.isclose(summary['llm_as_judge'], correct / len(answers), abs_tol=1e-12)
                or summary['answers_sha256'] != sha256(folder / 'answers.jsonl.gz')
                or summary['judge_results_sha256'] != sha256(judged / 'judgments.jsonl')
                or done['summary_sha256'] != sha256(judged / 'summary.json')
                or done['judgments_sha256'] != sha256(judged / 'judgments.jsonl')):
            raise ValueError('Judge coverage/hash audit failed')
        variants[name] = dict(count=len(answers), f1=sum(a['f1'] for a in answers) / len(answers),
                             bleu1=sum(a['bleu1'] for a in answers) / len(answers),
                             llm_as_judge=correct / len(answers),
                             finish_rate=sum(a['finished'] for a in answers) / len(answers))
    write_jsonl(output / 'paired-answers.jsonl.gz', paired.values())
    result = {'variants': variants, 'judge_protocol': config['study']['judge_protocol'],
              'delta_percentage_points': {k: 100 * (variants['construction_retrieval'][k] - variants['construction_only'][k])
                                         for k in ('f1', 'bleu1', 'llm_as_judge', 'finish_rate')},
              'construction_model': str(output / 'construction/final'),
              'shared_memory_sha256': sha256(output / 'memory-test/index.json')}
    write_json(output / 'comparison.json', result)
    lines = ['# Mem-T正式构建训练与组件评测', '',
             '两组使用同一份C200构建权重、同一份完整测试记忆及相同问题顺序；只更换检索权重。',
             f"测试题数：{len(questions)}。Judge协议：{config['study']['judge_protocol']}（Memory-R2宽松主题匹配）。", '',
             '| 指标 | 仅构建训练 C200+R0 | 构建与检索训练 C200+R200 | 差值（百分点） |',
             '|---|---:|---:|---:|']
    for key in ('f1', 'bleu1', 'llm_as_judge', 'finish_rate'):
        lines.append(f"| {key} | {100*variants['construction_only'][key]:.4f}% | {100*variants['construction_retrieval'][key]:.4f}% | {result['delta_percentage_points'][key]:+.4f} |")
    lines += ['', '构建训练：原始Qwen3-4B初始化，200步、全局batch32、学习率5e-6、cosine、warmup0.1、seed42。',
              '仅最终HF导出，不保存真实4B优化器状态；意外中断SFT需要从初始权重重新训练。',
              '此对照衡量C200固定时检索训练的增益；不能单独归因构建训练相对C0的增益。',
              'Judge采用宽松原文，主题相关而数值不同也可能判对；不能将该分数解释为严格事实准确率。',
              '采用新CrtRaw协议；旧协议历史分数单独保留，不直接混合比较。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result


def calibration_inputs(output):
    cases = [('Which city?', 'Paris', 'Paris', 'CORRECT'),
             ('When?', 'May 7', '7 May', 'CORRECT'),
             ('Which animal?', 'elephant', 'a teaspoon', 'WRONG'),
             ('How long?', '4 months and 10 days', 'six months', None)]
    rows, questions = [], []
    for i, (question, gold, pred, expected) in enumerate(cases):
        qa = dict(sample_id='judge-calibration', qa_id=str(i), category=1, question=question, answer=gold)
        questions.append(qa)
        rows.append(dict(qa, gold=gold, pred=pred, response=pred, expected_label=expected))
    ensure_rows(output / 'questions.jsonl', questions)
    ensure_rows(output / 'answers.jsonl.gz', rows)


def collector_config(config):
    result = copy.deepcopy(config)
    count = config.get('study', {}).get('collector_gpus', 4)
    if type(count) is not int or count not in (1, 4):
        raise ValueError('collector_gpus must be 1 or 4')
    result['distributed'].update(gpus=count, tensor_parallel=count, inference_tensor_parallel=count)
    if count == 1:
        result['distributed']['gpu_memory_utilization'] = 0.8
    return result


def run(config, args):
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root or os.getuid() != 2086:
        raise ValueError('Execute inside your own remote Mem-T project')
    output = Path(config['experiment']['output']).resolve()
    config['storage']['ledger'] = str(output / 'storage-ledger.json')
    stages = Stages(config, root, output, resume=args.resume)
    stages.check_budget('formal construction preflight')
    # Additional compressed data, final HF export, test records, runtime reserve.
    admission = output / 'ADMISSION.json'
    if not admission.exists():
        used = stages.budget.check(config['study']['peak_additional_bytes'], 'formal export-only peak admission')
        write_json(admission, {'baseline_bytes': used, 'additional_bytes': config['study']['peak_additional_bytes'],
                              'limit_bytes': config['storage']['limit_bytes']})
    if args.preflight:
        stages.state('preflight_complete')
        return
    if (config['construction']['protocol'] != 'paper-v2' or config['construction']['steps'] != 200
            or config['construction']['checkpoint_policy'] != 'export_only'
            or config['construction']['global_batch_size'] != 32
            or config['models']['construction'] != config['models']['base']):
        raise ValueError('Formal construction contract requires BASE initialization, paper-v2, 200 steps, batch32, export_only')
    allocated_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    collect_config = collector_config(config)
    if allocated_gpus not in (4, collect_config['distributed']['gpus']):
        raise ValueError('Allocation does not match collector or four-GPU SFT')
    if not os.getenv('SLURM_JOB_END_TIME'):
        allocation = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '-o'], text=True)
        end = re.search(r'\bEndTime=(\S+)', allocation).group(1)
        os.environ['SLURM_JOB_END_TIME'] = str(datetime.datetime.fromisoformat(end).timestamp())
    base, retr = config['models']['base'], config['models']['retrieval']
    judge_env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    calibration = output / 'calibration-inputs'
    calibration_inputs(calibration)
    def judge(name, answers, questions):
        folder = output / name
        command = [sys.executable, '-u', '-m', 'evaluation.judge', '--model', config['models']['judge'],
                   '--answers', answers, '--questions', questions, '--output', folder,
                   '--protocol', config['study']['judge_protocol']]
        stages.run(name, command, folder, inputs=[answers, questions], env=judge_env, resumable=True)
    judge('judge-calibration', calibration / 'answers.jsonl.gz', calibration / 'questions.jsonl')
    actual = read_jsonl(output / 'judge-calibration/judgments.jsonl')
    cases = read_jsonl(calibration / 'answers.jsonl.gz')
    for case, row in zip(cases, actual):
        if case['expected_label'] and case['expected_label'] != row['judge_label']:
            raise ValueError('Judge calibration failed on an unambiguous control')
    write_json(output / 'JUDGE_CALIBRATION.json', {'count': len(actual), 'basic_controls_passed': True,
               'duration_mismatch_label': actual[-1]['judge_label'], 'excluded_from_test_denominator': True,
               'protocol': config['study']['judge_protocol']})
    data = Path(config['experiment']['data'])
    train = data_view(data / 'train', output / 'data/train', config['experiment']['questions']['train'], config['seed'])
    test = data_view(data / 'test', output / 'data/test', config['experiment']['questions']['evaluation'], config['seed'])
    if set(read_json(train / 'partition.json')['sample_ids']) & set(read_json(test / 'partition.json')['sample_ids']):
        raise ValueError('Training/test conversations overlap')
    memory = output / 'memory-train'
    command = stages.cli('build-memory', config, memory, '--model', base, '--inference-engine', 'vllm',
                         '--data', train, '--resume', name='build-train')
    stages.run('build-train', command, memory, inputs=[train / 'episodes.jsonl', train / 'questions.jsonl'], resumable=True)
    forests = output / 'forests'
    command = stages.cli('collect-trees', collect_config, forests, '--model', retr, '--inference-engine', 'vllm',
                         '--memory', memory, '--resume', name='collect-forests')
    stages.run('collect-forests', command, forests, inputs=[memory / 'index.json', memory / 'questions.jsonl'], resumable=True)
    selected = output / 'selected'
    command = stages.cli('derive-sft', config, selected, '--memory', memory, '--forests', forests / 'forests.jsonl.gz', name='derive-sft')
    stages.run('derive-sft', command, selected, inputs=[forests / 'forests.jsonl.gz', memory / 'operations.jsonl.gz'])
    audit = read_json(selected / 'selection_audit.json')
    scored = read_jsonl(selected / 'scored.jsonl.gz')
    valid = [r for r in scored if r['valid'] and r.get('completion_valid', True) and r['score'] is not None]
    if (audit['selected'] < config['experiment']['target_selected_operations'] or not valid
            or len({r['score'] for r in valid}) < 2 or not any(r['trace_hits'] for r in valid)):
        raise ValueError('Insufficient genuine actions or degenerate attribution; no SFT started')
    del scored, valid
    if allocated_gpus != 4:
        stages.state('sft', deferred=True, reason='collector complete; four GPUs required for SFT')
        raise StageDeferred('sft: switch single-GPU collector allocation to four GPUs')
    construction = output / 'construction'
    command = stages.cli('train-sft', config, construction, '--model', base, '--selected', selected / 'selected.jsonl.gz', name='sft')
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', '-m', 'cli', *command[4:]]
    stages.run('sft', command, construction, inputs=[selected / 'selected.jsonl.gz'],
               minimum_seconds=config['study']['sft_minimum_job_seconds'])
    trained = construction / 'final'
    stages.run('verify-export', [sys.executable, '-u', '-m', 'evaluation.verify_model', '--config', args.config,
               '--model', trained, '--reference', base, '--output', output / 'verify-export'], output / 'verify-export',
               inputs=[construction / 'SFT_COMPLETE.json'])
    cfg = copy.deepcopy(config)
    cfg['models']['construction'] = str(trained)
    cfg['construction'].update(rollouts=1, minimum_valid_operations=0)
    memory = output / 'memory-test'
    command = stages.cli('build-memory', cfg, memory, '--model', trained, '--inference-engine', 'vllm',
                         '--data', test, '--resume', name='build-test')
    stages.run('build-test', command, memory, inputs=[test / 'episodes.jsonl', test / 'questions.jsonl'], resumable=True)
    for name, model in [('construction_only', base), ('construction_retrieval', retr)]:
        folder = output / ('evaluate-' + name)
        command = stages.cli('evaluate', cfg, folder, '--model', model, '--inference-engine', 'vllm',
                             '--memory', memory, '--resume', name='evaluate-' + name)
        stages.run('evaluate-' + name, command, folder, inputs=[memory / 'index.json', memory / 'questions.jsonl'], resumable=True)
        judge('judge-' + name, folder / 'answers.jsonl.gz', memory / 'questions.jsonl')
    result = compare(output, config)
    sft = read_json(construction / 'SFT_COMPLETE.json')
    proofs = [read_json(construction / f'parameter-proof-rank-{i}.json') for i in range(4)]
    if sft['step'] != 200 or sft['global_batch_size'] != 32 or not all(p['changed'] and p['completed_step'] == 200 for p in proofs):
        raise ValueError('Real four-rank SFT update proof incomplete')
    if list(construction.glob('checkpoint-*')):
        raise ValueError('Unexpected optimizer checkpoint under export-only policy')
    for role, expected in stages.contract['models'].items():
        if model_identity(config['models'][role]) != expected:
            raise ValueError('An input model changed: ' + role)
    used = stages.check_budget('formal construction complete')
    write_json(output / 'COMPLETE.json', {'completed_unix': time.time(), 'result': result,
               'comparison_sha256': sha256(output / 'comparison.json'), 'report_sha256': sha256(output / 'REPORT.md'),
               'storage_bytes': used, 'shared_cache_bytes': stages.shared_cache['observed_bytes'],
               'charged_including_shared_bytes': used + stages.shared_cache['observed_bytes'],
               'full_4b_resume_verified': False, 'input_models_preserved': True})
    stages.state('complete')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        run(config, args)
    except StageDeferred as error:
        write_json(Path(config['experiment']['output']) / 'DEFERRED.json',
                   {'job': os.environ['SLURM_JOB_ID'], 'reason': str(error), 'time_unix': time.time()})
        raise SystemExit(75)


if __name__ == '__main__':
    main()
