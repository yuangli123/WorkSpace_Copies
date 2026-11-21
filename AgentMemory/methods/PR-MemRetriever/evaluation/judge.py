"""Score the complete Mem-T test answers with the base Qwen3-4B judge."""
import collections
import gzip
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('sample_id', 'qa_id', 'category', 'question', 'gold', 'pred')
SCHEMA = {'type': 'object', 'properties': {'reasoning': {'type': 'string'},
          'label': {'type': 'string', 'enum': ['CORRECT', 'WRONG']}},
          'required': ['reasoning', 'label'], 'additionalProperties': False}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def verdict(text, require_reasoning=True):
    value = json.loads(text)
    assert isinstance(value, dict) and value.get('label') in ('CORRECT', 'WRONG')
    if require_reasoning:
        assert isinstance(value.get('reasoning'), str) and value['reasoning'].strip()
    elif 'reasoning' in value:
        assert isinstance(value['reasoning'], str)
    return value


def verify_prefix(saved, source):
    assert len(saved) <= len(source)
    for actual, expected in zip(saved, source):
        assert all(actual[k] == expected[k] for k in FIELDS), 'Saved judgments do not match ordered test answers'
        assert actual['judge_label'] in ('CORRECT', 'WRONG')


def self_test():
    assert verdict('{"reasoning":"Same answer.","label":"CORRECT"}')['label'] == 'CORRECT'
    for text in ('not json', '{"label":"MAYBE","reasoning":"x"}', '{"label":"WRONG"}'):
        try:
            verdict(text)
        except (AssertionError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError('Invalid judgment accepted')
    a = dict(sample_id='same-conversation', qa_id='q1', category=1, question='a', gold='b', pred='c')
    b = dict(a, qa_id='q2')
    verify_prefix([dict(a, judge_label='CORRECT')], [a, b])
    try:
        verify_prefix([dict(b, judge_label='CORRECT')], [a, b])
    except AssertionError:
        pass
    else:
        raise AssertionError('Conversation ID collision accepted')
    print('Judge parser and ordered full-coverage resume self-checks passed')


def main(argv=None):
    import fcntl
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('answers', 'questions', 'output', 'model'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--protocol', choices=['legacy-topic-v1', 'memory-r2-generous-v1'], default='legacy-topic-v1')
    args = parser.parse_args(argv)
    assert Path.cwd().resolve() == ROOT
    output, MODEL = Path(args.output).resolve(), Path(args.model).resolve()
    assert ROOT in output.parents and not output.is_symlink()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'JUDGE.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_file = Path(args.answers).resolve()
    source_sha = sha(source_file)
    with gzip.open(source_file, 'rt', encoding='utf-8') as stream:
        source = [json.loads(line) for line in stream if line.strip()]
    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    count = len(questions)
    assert count > 0 and len(source) == count
    assert len({(row['sample_id'], row['qa_id']) for row in source}) == count
    for row, qa in zip(source, questions):
        assert all(row[key] == qa[key] for key in ('sample_id', 'qa_id', 'category', 'question', 'answer'))
        assert row['gold'] == row['answer'] and row['pred'] == row['response']
        assert isinstance(row['gold'], str) and isinstance(row['pred'], str)
    from evaluation.prompts import judge_prompts
    system, template = judge_prompts(args.protocol)
    schema = dict(SCHEMA, required=['label']) if args.protocol == 'memory-r2-generous-v1' else SCHEMA
    prompt_source = ROOT / 'evaluation/prompts.py'
    constants = {'LOCOMO_SYSTEM_PROMPT': system, 'LOCOMO_USER_PROMPT_TEMPLATE': template}
    weights = {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(MODEL.glob('*.safetensors'))}
    assert weights
    spec = {'judge_model': str(MODEL), 'model_config_sha256': sha(MODEL / 'config.json'), 'model_weights': weights,
            'answers_sha256': source_sha, 'questions_sha256': sha(Path(args.questions)),
            'judge_prompt_source_sha256': sha(prompt_source), 'judge_script_sha256': sha(Path(__file__)),
            'protocol': args.protocol, 'prompts': constants, 'schema': schema, 'seed': 42, 'temperature': 0,
            'enable_thinking': False, 'max_model_len': 8192, 'max_tokens': 512, 'retry_max_tokens': 1024,
            'request_batch': 32, 'guided_decoding_backend': 'xgrammar', 'tensor_parallel_size': 1,
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'vllm', 'xgrammar')},
            'score_definition': f'CORRECT / all {count} answers; no skipped or error rows permitted'}
    manifest_file = output / 'manifest.json'
    if manifest_file.exists():
        assert json.loads(manifest_file.read_text())['spec'] == spec, 'Judge protocol or input changed'
    else:
        write_json(manifest_file, {'spec': spec, 'created_unix': time.time()})
    results_file = output / 'judgments.jsonl'
    saved = [json.loads(line) for line in results_file.read_text().splitlines() if line.strip()] if results_file.exists() else []
    verify_prefix(saved, source)
    started = time.time()
    write_json(output / 'RUNNING.json', {'pid': os.getpid(), 'job': os.environ.get('SLURM_JOB_ID'),
               'started_unix': started, 'completed': len(saved), 'target': len(source)})
    if len(saved) < len(source):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
        prompts = []
        for row in source:
            messages = [{'role': 'system', 'content': constants['LOCOMO_SYSTEM_PROMPT']},
                        {'role': 'user', 'content': constants['LOCOMO_USER_PROMPT_TEMPLATE'].format(
                            question=row['question'], gold_answer=row['gold'], generated_answer=row['pred'])}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert len(ids) + 1024 <= 8192, 'Judge prompt would be truncated'
            prompts.append(ids)
        write_json(output / 'token_lengths.json', {'count': len(prompts), 'max_prompt_tokens': max(map(len, prompts)),
                   'min_prompt_tokens': min(map(len, prompts)), 'truncated': 0})
        engine = LLM(model=str(MODEL), tokenizer=str(MODEL), trust_remote_code=False, dtype='bfloat16',
                     tensor_parallel_size=1, gpu_memory_utilization=0.60, max_model_len=8192,
                     max_num_seqs=32, max_num_batched_tokens=8192, enforce_eager=True, seed=42,
                     guided_decoding_backend='xgrammar')
        with results_file.open('a', encoding='utf-8') as stream:
            for offset in range(len(saved), len(source), 32):
                batch = source[offset:offset + 32]
                pending = list(range(len(batch)))
                records = {}
                for attempt, max_tokens in enumerate((512, 1024), 1):
                    if not pending:
                        break
                    parameters = SamplingParams(temperature=0, seed=42, max_tokens=max_tokens,
                                 guided_decoding=GuidedDecodingParams(json=schema))
                    completions = engine.generate(prompt_token_ids=[prompts[offset + i] for i in pending],
                                                  sampling_params=parameters, use_tqdm=False)
                    assert len(completions) == len(pending)
                    failed = []
                    for index, completion in zip(pending, completions):
                        text = completion.outputs[0].text
                        try:
                            value = verdict(text, require_reasoning=args.protocol == 'legacy-topic-v1')
                            assert completion.outputs[0].finish_reason != 'length'
                        except (AssertionError, json.JSONDecodeError) as error:
                            with (output / 'format-errors.jsonl').open('a') as errors:
                                errors.write(json.dumps({'index': offset + index, 'attempt': attempt,
                                             'raw_output': text, 'error': str(error)}) + '\n')
                            failed.append(index)
                            continue
                        records[index] = {key: batch[index][key] for key in FIELDS}
                        records[index].update(judge_label=value['label'], judge_reasoning=value.get('reasoning', ''),
                                              judge_raw_output=text, attempt=attempt,
                                              finish_reason=completion.outputs[0].finish_reason,
                                              prompt_tokens=len(prompts[offset + index]),
                                              output_tokens=len(completion.outputs[0].token_ids))
                    pending = failed
                assert not pending, 'Unresolved judge formatting errors; no incomplete score is published'
                for index in range(len(batch)):
                    record = records[index]
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    saved.append(record)
                stream.flush()
                os.fsync(stream.fileno())
                correct = sum(row['judge_label'] == 'CORRECT' for row in saved)
                progress = {'time_unix': time.time(), 'completed': len(saved), 'target': len(source),
                            'correct_so_far': correct, 'last_qa_id': saved[-1]['qa_id']}
                write_json(output / 'PROGRESS.json', progress)
                print(json.dumps(progress), flush=True)
    verify_prefix(saved, source)
    assert len(saved) == count and sha(source_file) == source_sha
    assert weights == {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(MODEL.glob('*.safetensors'))}
    counts = collections.Counter(row['judge_label'] for row in saved)
    categories = {}
    for category in sorted({str(row['category']) for row in saved}):
        subset = [row for row in saved if str(row['category']) == category]
        correct = sum(row['judge_label'] == 'CORRECT' for row in subset)
        categories[category] = {'count': len(subset), 'correct': correct, 'wrong': len(subset) - correct,
                                'llm_as_judge': correct / len(subset)}
    summary = {'judge_model': str(MODEL), 'protocol': args.protocol, 'count': len(saved), 'correct': counts['CORRECT'], 'wrong': counts['WRONG'],
               'error': 0, 'skipped': 0, 'llm_as_judge': counts['CORRECT'] / len(saved), 'by_category': categories,
               'format_retried_questions': sum(row['attempt'] > 1 for row in saved), 'wall_seconds_this_invocation': time.time() - started,
               'answers_sha256': source_sha, 'judge_results_sha256': sha(results_file), 'manifest_sha256': sha(manifest_file)}
    write_json(output / 'summary.json', summary)
    report = ['# Qwen3-4B LLM-as-judge', '', f"完整测试：{len(saved)}题。CORRECT={counts['CORRECT']}，WRONG={counts['WRONG']}。",
              f"LLM-as-judge = {summary['llm_as_judge']:.6f} ({100 * summary['llm_as_judge']:.2f}%)。", '',
              f'使用基础Qwen3-4B；提示词版本：{args.protocol}；宽松主题匹配，temperature=0，关闭thinking，JSON约束输出。',
              f'分母为全部{count}题，逐题输出及完整协议见judgments.jsonl和manifest.json。',
              '这是Qwen3-4B评审模型的分数，与其他评审模型的分数应分别报告。', '', '| category | n | correct | score |', '|---|---:|---:|---:|']
    report += [f"| {k} | {v['count']} | {v['correct']} | {v['llm_as_judge']:.6f} |" for k, v in categories.items()]
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n')
    write_json(output / 'COMPLETE.json', {'completed_unix': time.time(), 'count': count,
               'summary_sha256': sha(output / 'summary.json'), 'judgments_sha256': sha(results_file)})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    else:
        main()
