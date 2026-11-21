"""Budgeted construction training and a same-protocol held-out comparison."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

from .io import load_config, read_json, read_jsonl, sha256, write_json, write_jsonl
from .stages import Stages, model_identity


def subset(questions, count, seed):
    if count >= len(questions):
        return list(questions)
    groups = {}
    for i, row in enumerate(questions):
        groups.setdefault(str(row['category']), []).append(i)
    rng = random.Random(seed)
    for indices in groups.values():
        rng.shuffle(indices)
    chosen = []
    while len(chosen) < count:
        for category in sorted(groups):
            if groups[category] and len(chosen) < count:
                chosen.append(groups[category].pop())
        if not any(groups.values()) and len(chosen) < count:
            raise ValueError('Insufficient questions')
    return [questions[i] for i in sorted(chosen)]


def ensure_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Prepared input changed: ' + str(path))
    else:
        write_json(path, value)


def ensure_rows(path, value):
    if path.exists():
        if read_jsonl(path) != value:
            raise ValueError('Prepared rows changed: ' + str(path))
    else:
        write_jsonl(path, value)


def data_view(source, output, count, seed):
    episodes = read_jsonl(source / 'episodes.jsonl')
    questions = subset(read_jsonl(source / 'questions.jsonl'), count, seed)
    partition = read_json(source / 'partition.json')
    partition.update(scope='construction-smoke' if count < len(read_jsonl(source / 'questions.jsonl')) else 'full',
                     parent_questions_sha256=sha256(source / 'questions.jsonl'))
    ensure_rows(output / 'episodes.jsonl', episodes)
    ensure_rows(output / 'questions.jsonl', questions)
    ensure_json(output / 'partition.json', partition)
    return output


def full_memory_view(source, output, full_questions):
    index = read_json(source / 'index.json')
    for name in ['index.json', 'partition.json', 'manifest.json', 'operations.jsonl.gz'] + [v['snapshot'] for v in index.values()]:
        src, dst = source / name, output / name
        if dst.exists():
            if sha256(src) != sha256(dst):
                raise ValueError('Memory view content changed')
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    questions = []
    for sid, entry in index.items():
        replica = sid.rsplit('__r', 1)[1]
        questions.extend(dict(q, sample_id=sid, qa_id=q['qa_id'] + '__r' + replica)
                         for q in full_questions if q['sample_id'] == entry['source_sample_id'])
    ensure_rows(output / 'questions.jsonl', questions)
    complete = read_json(source / 'COMPLETE.json')
    complete['questions'] = len(questions)
    ensure_json(output / 'COMPLETE.json', complete)
    return output


def attribution_usable(folder):
    rows = read_jsonl(folder / 'scored.jsonl')
    valid = [r for r in rows if r['valid'] and r.get('completion_valid', True) and r['score'] is not None]
    return bool(valid and any(abs(r['score']) > 1e-12 for r in valid)
                and len({r['score'] for r in valid}) > 1 and any(r['trace_hits'] for r in valid))


def compare(output, config):
    from methods.retrieval.tree import answer_f1, answer_bleu1
    variants = {}
    aligned = None
    evaluation_contract = None
    for name in ('base', 'trained'):
        folder = output / ('evaluate-' + name)
        answers = read_jsonl(folder / 'answers.jsonl.gz')
        questions = read_jsonl(output / ('memory-eval-' + name) / 'questions.jsonl')
        judgments = read_jsonl(output / ('judge-' + name) / 'judgments.jsonl')
        if (len(answers) != len(questions) or len(judgments) != len(answers)
                or len(answers) != config['experiment']['questions']['evaluation']):
            raise ValueError('Incomplete held-out comparison')
        recorded = read_json(folder / 'manifest.json')
        normalized = copy.deepcopy(recorded['config'])
        if (normalized['models']['retrieval'] != config['models']['retrieval']
                or normalized['model']['checkpoint'] != str(Path(config['models']['retrieval']).resolve())
                or normalized['construction']['protocol'] != 'paper-v2'):
            raise ValueError('Retrieval role or construction protocol changed')
        normalized['models'].pop('construction')
        normalized['storage'].pop('ledger')
        contract = {'config': normalized, 'code': recorded['code_sha256'], 'packages': recorded['packages']}
        if evaluation_contract is not None and evaluation_contract != contract:
            raise ValueError('Evaluation settings/source differ across construction variants')
        evaluation_contract = contract
        keys = [(r['sample_id'], r['qa_id'], r['question'], r['answer'], r['category']) for r in answers]
        if len(set((r['sample_id'], r['qa_id']) for r in answers)) != len(answers):
            raise ValueError('Duplicate held-out answers')
        if aligned is not None and keys != aligned:
            raise ValueError('Comparison questions/order changed')
        aligned = keys
        for row, qa, judge in zip(answers, questions, judgments):
            if any(row[k] != qa[k] for k in ('sample_id', 'qa_id', 'question', 'answer', 'category')):
                raise ValueError('Answer does not match held-out question')
            if any(judge[k] != row[k] for k in ('sample_id', 'qa_id', 'question', 'gold', 'pred', 'category')):
                raise ValueError('Judge row does not match answer')
            verdict = json.loads(judge['judge_raw_output'])
            if (verdict['label'] != judge['judge_label'] or verdict['label'] not in ('CORRECT', 'WRONG')
                    or verdict['reasoning'] != judge['judge_reasoning'] or not verdict['reasoning'].strip()
                    or judge['finish_reason'] == 'length'):
                raise ValueError('Invalid judge verdict')
            for field, fn in [('f1', answer_f1), ('bleu1', answer_bleu1)]:
                if not math.isclose(row[field], fn(row['pred'], row['gold']), abs_tol=1e-12):
                    raise ValueError('Answer metric does not recompute')
        summary = read_json(output / ('judge-' + name) / 'summary.json')
        if summary['count'] != len(answers) or summary['error'] or summary['skipped']:
            raise ValueError('Judge denominator or error count invalid')
        correct = sum(r['judge_label'] == 'CORRECT' for r in judgments)
        judge_folder = output / ('judge-' + name)
        done = read_json(judge_folder / 'COMPLETE.json')
        if (summary['correct'] != correct or summary['wrong'] != len(answers)-correct
                or not math.isclose(summary['llm_as_judge'], correct/len(answers), abs_tol=1e-12)
                or summary['answers_sha256'] != sha256(folder / 'answers.jsonl.gz')
                or summary['judge_results_sha256'] != sha256(judge_folder / 'judgments.jsonl')
                or done['summary_sha256'] != sha256(judge_folder / 'summary.json')
                or done['judgments_sha256'] != sha256(judge_folder / 'judgments.jsonl')):
            raise ValueError('Judge score/input/completion hash does not verify')
        variants[name] = {'count': len(answers), 'f1': sum(r['f1'] for r in answers) / len(answers),
                          'bleu1': sum(r['bleu1'] for r in answers) / len(answers),
                          'llm_as_judge': sum(r['judge_label'] == 'CORRECT' for r in judgments) / len(answers),
                          'finish_rate': sum(r['finished'] for r in answers) / len(answers),
                          'answers_sha256': sha256(folder / 'answers.jsonl.gz')}
    specs = [read_json(output / ('judge-' + name) / 'manifest.json')['spec'] for name in ('base', 'trained')]
    for spec in specs:
        spec.pop('answers_sha256')
    if specs[0] != specs[1]:
        raise ValueError('Judge protocol differs across construction variants')
    result = {'protocol': config['construction']['protocol'], 'variants': variants,
              'delta_percentage_points': {k: 100*(variants['trained'][k]-variants['base'][k])
                                         for k in ('f1', 'bleu1', 'llm_as_judge', 'finish_rate')},
              'scope': config['experiment']['scope'], 'evaluation_split': config['experiment']['evaluation_split']}
    write_json(output / 'comparison.json', result)
    lines = ['# Mem-T构建训练对照与验收', '', f"协议：paper-v2；题数：{variants['base']['count']}；划分：{config['experiment']['evaluation_split']}。检索器在两组间保持冻结。", '',
             '| 指标 | 基础构建模型 | 训练后构建模型 | 差值（百分点） |', '|---|---:|---:|---:|']
    for key in ('f1', 'bleu1', 'llm_as_judge', 'finish_rate'):
        lines.append(f"| {key} | {100*variants['base'][key]:.4f}% | {100*variants['trained'][key]:.4f}% | {result['delta_percentage_points'][key]:+.4f} |")
    lines += ['', f"构建SFT：Qwen3-4B、4卡、全局batch32、{config['construction']['steps']}步；检查点策略：{config['construction']['checkpoint_policy']}。",
              '完整保存/恢复与连续更新对照仅由微型四卡Qwen3验证，见tiny-resume/VERIFIED.json。',
              '短程模式仅为机制与运行验收，不要求4步后提升，也不代表完成10k操作或1307题正式实验。',
              '主后端为Trainer+ZeRO-3原子动作mask，区别于论文的LLaMA-Factory框架；未恢复作者未公开细节。',
              '新协议新增可学习CrtRaw；旧协议下历史F1不作为本表同条件基线。']
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root or os.getuid() != 2086:
        raise ValueError('Execute inside your own remote Mem-T project')
    output = Path(config['experiment']['output']).resolve()
    config['storage']['ledger'] = str(output / 'storage-ledger.json')
    if config['construction']['protocol'] != 'paper-v2':
        raise ValueError('Construction study requires explicit paper-v2 protocol')
    if config['construction']['checkpoint_policy'] != 'export_only' and not args.preflight:
        # Formal full-state profiles must pass admission before expensive sampling.
        from .storage import DiskBudget
        nbytes = sum(p.stat().st_size for p in Path(config['models']['base']).glob('*.safetensors'))
        DiskBudget(config['storage']).check(16*nbytes, 'formal full-state SFT peak admission')
    stages = Stages(config, root, output, resume=args.resume)
    if args.preflight:
        stages.state('preflight_complete', storage_bytes=stages.check_budget('preflight complete'))
        return
    if len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) != 4:
        raise ValueError('This acceptance pipeline requires four allocated GPUs')
    data = Path(config['experiment']['data'])
    counts = config['experiment']['questions']
    train_data = data_view(data / 'train', output / 'data/train', counts['train'], config['seed'])
    split = config['experiment']['evaluation_split']
    if split not in ('valid', 'test'):
        raise ValueError('Evaluation must use a held-out split')
    valid_data = data_view(data / split, output / 'data/evaluation', counts['evaluation'], config['seed'])
    if set(read_json(train_data/'partition.json')['sample_ids']) & set(read_json(valid_data/'partition.json')['sample_ids']):
        raise ValueError('Train/validation conversation overlap')
    base, retr = config['models']['base'], config['models']['retrieval']
    stages.run('tiny-resume', [sys.executable, '-u', '-m', 'experiments.tiny_resume', '--config', args.config,
               '--output', output/'tiny-resume'], output/'tiny-resume', inputs=[Path(args.config)])
    memory = output / 'memory-train'
    command = stages.cli('build-memory', config, memory, '--model', base, '--inference-engine', 'vllm',
                         '--data', train_data, '--resume', name='build-train')
    stages.run('build-train', command, memory, inputs=[train_data/'episodes.jsonl', train_data/'questions.jsonl'], resumable=True)
    selected = None
    for label, candidate in [('initial', memory), ('expanded', None)]:
        if candidate is None:
            candidate = full_memory_view(memory, output/'memory-train-expanded', read_jsonl(data/'train/questions.jsonl'))
        forests, chosen = output/('forests-'+label), output/('selected-'+label)
        command = stages.cli('collect-trees', config, forests, '--model', retr, '--inference-engine', 'vllm',
                             '--memory', candidate, '--resume', name='forests-'+label)
        stages.run('forests-'+label, command, forests, inputs=[candidate/'index.json',candidate/'questions.jsonl'], resumable=True)
        command = stages.cli('derive-sft', config, chosen, '--memory', candidate, '--forests', forests/'forests.jsonl', name='select-'+label)
        stages.run('select-'+label, command, chosen, inputs=[forests/'forests.jsonl', candidate/'operations.jsonl.gz'])
        if attribution_usable(chosen):
            selected = chosen / 'selected.jsonl'
            break
    if selected is None:
        stages.state('attribution_degenerate', failed=True)
        raise ValueError('No non-degenerate attribution even after all training questions; not training on tied/zero credit')
    target = config['experiment'].get('target_selected_operations', 0)
    if len(read_jsonl(selected)) < target:
        raise ValueError('Insufficient genuine selected operations for formal profile; increase independent construction rollouts in a new run')
    construction = output / 'construction'
    command = stages.cli('train-sft', config, construction, '--model', base, '--selected', selected, name='sft')
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4', '-m', 'cli', *command[4:]]
    stages.run('sft', command, construction, inputs=[selected])
    trained = construction / 'final'
    stages.run('verify-export', [sys.executable, '-u', '-m', 'evaluation.verify_model', '--config', args.config,
               '--model', trained, '--reference', base, '--output', output/'verify-export'], output/'verify-export',
               inputs=[construction/'SFT_COMPLETE.json'])
    for variant, model in [('base', base), ('trained', trained)]:
        cfg = copy.deepcopy(config)
        cfg['models']['construction'] = str(model)
        folder = output / ('memory-eval-'+variant)
        command = stages.cli('build-memory', cfg, folder, '--model', model, '--inference-engine', 'vllm',
                             '--data', valid_data, '--resume', name='build-eval-'+variant)
        stages.run('build-eval-'+variant, command, folder, inputs=[valid_data/'episodes.jsonl', valid_data/'questions.jsonl'], resumable=True)
        evaluation = output / ('evaluate-'+variant)
        command = stages.cli('evaluate', cfg, evaluation, '--model', retr, '--inference-engine', 'vllm', '--memory', folder, name='evaluate-'+variant)
        stages.run('evaluate-'+variant, command, evaluation, inputs=[folder/'index.json',folder/'questions.jsonl'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
        judged = output / ('judge-'+variant)
        command = [sys.executable, '-u', '-m', 'evaluation.judge', '--model', config['models']['judge'],
                   '--answers', evaluation/'answers.jsonl.gz', '--questions', folder/'questions.jsonl', '--output', judged]
        stages.run('judge-'+variant, command, judged, inputs=[evaluation/'answers.jsonl.gz',folder/'questions.jsonl'], env=env, resumable=True)
    result = compare(output, config)
    sft = read_json(construction/'SFT_COMPLETE.json')
    proofs = [read_json(construction/f'parameter-proof-rank-{rank}.json') for rank in range(4)]
    if (sft['step'] != config['construction']['steps'] or sft['global_batch_size'] != 32
            or not all(p['completed_step'] == sft['step'] for p in proofs) or not any(p['changed'] for p in proofs)):
        raise ValueError('Four-rank real SFT update evidence is incomplete')
    if config['construction']['checkpoint_policy'] == 'export_only' and list(construction.glob('checkpoint-*')):
        raise ValueError('Export-only policy unexpectedly saved a large checkpoint')
    # Frozen role identities must still match after the entire experiment.
    for role, expected in stages.contract['models'].items():
        if model_identity(config['models'][role]) != expected:
            raise ValueError('An input model changed: '+role)
    used = stages.check_budget('construction acceptance complete')
    write_json(output / 'COMPLETE.json', {'completed_unix': time.time(), 'scope': config['experiment']['scope'],
               'comparison_sha256': sha256(output/'comparison.json'), 'report_sha256': sha256(output/'REPORT.md'),
               'stage_receipts_sha256': {p.name: sha256(p) for p in (output/'stages').glob('*.json')},
               'tiny_full_resume_verified': True, 'full_4b_resume_verified': False, 'storage_bytes': used,
               'shared_cache_bytes': stages.shared_cache['observed_bytes'],
               'charged_including_shared_bytes': used + stages.shared_cache['observed_bytes'],
               'models_preserved': True, 'result': result})
    stages.state('complete', storage_bytes=used)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
