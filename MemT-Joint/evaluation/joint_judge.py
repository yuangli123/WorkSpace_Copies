"""Offline judges. Gold is used here only, never by the policy/Answer runtime."""
import copy
import json
from pathlib import Path

from evaluation.prompts import judge_prompts
from evaluation.longmemeval_prompt import get_anscheck_prompt
from experiments.io import checked_output, read_json, read_jsonl, write_json, write_jsonl, sha256


def score(config, answers, output, protocol, model=None, resume=False):
    from models.hf import HFPolicy
    output = checked_output(output, resume)
    cfg = copy.deepcopy(config)
    cfg['model'].update(max_new_tokens=512)
    location = model or cfg['models']['base']
    contract = {'answers_sha256': sha256(answers), 'protocol': protocol, 'config': cfg,
                'model': str(Path(location).resolve()),
                'weights': {p.name: sha256(p) for p in Path(location).glob('*.safetensors')}}
    manifest = output / 'manifest.json'
    if manifest.exists() and read_json(manifest) != contract:
        raise ValueError('Judge resume inputs/model/protocol changed')
    write_json(manifest, contract)
    rows = read_jsonl(answers)
    if not rows or len({r['qa_id'] for r in rows}) != len(rows):
        raise ValueError('Judge requires nonempty unique QA coverage')
    policy = HFPolicy(location, cfg)
    judgments = []
    for index, row in enumerate(rows):
        path = output / 'questions' / ('%06d.json' % index)
        if path.exists():
            saved = read_json(path)
            if saved['qa_id'] != row['qa_id']:
                raise ValueError('Judge saved prefix does not match answers')
            judgments.append(saved)
            continue
        if protocol == 'longmemeval-official-prompt':
            text = get_anscheck_prompt(row['category'], row['question'], row['answer'], row['response'], '_abs' in row['qa_id'])
            messages = [{'role': 'user', 'content': text}]
        else:
            system, template = judge_prompts(protocol)
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': template.format(
                question=row['question'], gold_answer=row['answer'], generated_answer=row['response'])}]
        completion = policy.complete(messages, temperature=0, stop_at_tool=False)
        raw = completion.text.strip()
        if protocol == 'longmemeval-official-prompt':
            label = raw.lower().rstrip('.')
            valid = label in ('yes', 'no')
            correct = label == 'yes'
        else:
            try:
                label = json.loads(raw)['label']
                valid, correct = label in ('CORRECT', 'WRONG'), label == 'CORRECT'
            except (json.JSONDecodeError, KeyError, TypeError):
                valid, correct = False, False
        if not valid:
            write_json(output / 'FORMAT_ERROR.json', {'qa_id': row['qa_id'], 'raw': raw})
            raise ValueError('Unresolved judge format; no incomplete Accuracy is published')
        result = {'qa_id': row['qa_id'], 'sample_id': row['sample_id'], 'label': label, 'correct': correct,
                  'raw': raw, 'prompt_tokens': len(completion.prompt_ids), 'generated_tokens': len(completion.response_ids)}
        write_json(path, result)
        judgments.append(result)
    summary = {'count': len(rows), 'score': sum(r['correct'] for r in judgments) / len(rows), 'protocol': protocol,
               'model': str(location), 'metric': 'Accuracy' if protocol == 'longmemeval-official-prompt' else 'LLM-as-a-judge',
               'parser': 'strict complete yes/no' if protocol == 'longmemeval-official-prompt' else 'strict JSON label',
               'note': 'Official prompt; report judge model and parser explicitly. F1 was not renamed to Accuracy.'}
    write_jsonl(output / 'judgments.jsonl', judgments)
    write_json(output / 'summary.json', summary)
    write_json(output / 'COMPLETE.json', {'count': len(rows), 'judgments_sha256': sha256(output / 'judgments.jsonl')})
    return summary
