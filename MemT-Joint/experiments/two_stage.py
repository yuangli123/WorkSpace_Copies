"""M6 collection: fixed trained R -> hindsight selection -> existing atomic SFT."""
from __future__ import annotations

import copy
import re
import time

from experiments.io import checked_output, write_json, write_jsonl, read_json, sha256
from experiments.joint import corpus, make_runtime, serializable, identity
from memory.branch import State, digest
from methods.joint.rollout import blocks, formation, evolution, new_state
from methods.joint.retrieval import sample_many
from methods.joint.credit import score_record
from methods.construction.credit import hindsight_scores, select_operations


def atomic_examples(rows, operations, sample_id, phase):
    result = []
    for row in rows:
        if not row['valid']:
            continue
        spans = list(re.finditer(r'<tool_call>.*?</tool_call>', row['text'], re.S))
        own = [op for op in operations if op['op_id'] in {digest([row['event'], i]) for i in range(len(spans))}]
        if len(own) != len(spans):
            raise ValueError('Atomic operation/completion mismatch')
        for index, op in enumerate(own):
            result.append(dict(op, sample_id=sample_id, phase=phase, category=op['call']['name'], valid=True,
                               completion_valid=True, messages=row['messages'], response=row['text'],
                               target_span=[0 if index == 0 else spans[index - 1].end(), spans[index].end()],
                               completion_id=digest(row['event']), protocol_version='paper-v2'))
    return result


def collect_sft(config, data, retrieval_model, output):
    cfg = copy.deepcopy(config)
    cfg.update(backend='hf')
    cfg['experiment'].update(variant='M6-collection', train_roles=['B', 'U', 'R'],
                              components={'construction': cfg['models']['base'], 'retrieval': retrieval_model})
    output = checked_output(output)
    from pathlib import Path
    if read_json(Path(data) / 'partition.json')['split'] != 'train':
        raise ValueError('Construction supervision may only use the train partition')
    contract = identity(cfg, data)
    contract['retrieval_weights'] = {p.name: sha256(p) for p in Path(retrieval_model).glob('*.safetensors')}
    write_json(output / 'manifest.json', contract)
    episodes, questions = corpus(data)
    runtime, _, _, _ = make_runtime(cfg, False)
    scored = []
    for replica in range(cfg['construction']['rollouts']):
        for episode in episodes:
            started = time.perf_counter()
            runtime.calls = []
            episode = copy.deepcopy(episode)
            source_id = episode['sample_id']
            sid = source_id + '__r' + str(replica)
            episode['sample_id'] = sid
            qs = [dict(q, sample_id=sid, qa_id=q['qa_id'] + '__r' + str(replica)) for q in questions
                  if q['sample_id'] == source_id and q.get('eval_session') is None]
            if not qs:
                raise ValueError('Two-stage hindsight collection requires terminal training questions')
            state, operations = new_state(sid, runtime), []
            for bi, block in enumerate(blocks(episode, cfg['memory']['summary_context_turns'])):
                state.observe(block.session, block.turns, block.position, runtime.count)
                event = 'M6/%s/block-%d' % (sid, bi)
                state, candidates, rows, ops = formation(state, block, runtime, event + '/B')
                operations += atomic_examples(rows, ops, sid, 'formation')
                state, rows, ops, _ = evolution(state, candidates, block, runtime, event + '/U')
                operations += atomic_examples(rows, ops, sid, 'evolution')
            forests = sample_many([state] * len(qs), qs, runtime, ['M6/' + q['qa_id'] for q in qs], forest=True)
            records = []
            for forest in forests:
                score_record(forest, runtime, mot=True)
                trees = []
                for tree in forest['trees']:
                    nodes = []
                    for node in tree:
                        retrieved = [{'operation_ids': state.journal.versions[ref]['operation_ids']}
                                     for ref in node.observed]
                        nodes.append({'children': node.children, 'advantage': node.advantage, 'retrieved': retrieved})
                    trees.append(nodes)
                records.append({'sample_id': sid, 'qa_id': forest['question']['qa_id'], 'trees': trees})
            scored += hindsight_scores(operations, records, qs, cfg)
            safe_name = digest(sid)[:24]
            write_json(output / 'forests' / (safe_name + '.json.gz'), serializable(forests))
            write_json(output / 'snapshots' / (safe_name + '.json'), state.to_dict())
            write_json(output / 'metrics' / (safe_name + '.json'), {'sample_id': sid, 'operations': len(operations),
                       'role_prompt_tokens': sum(len(r['prompt_ids']) for r in runtime.calls),
                       'role_generated_tokens': sum(len(r['response_ids']) for r in runtime.calls),
                       'answer_calls': 0, 'wall_seconds': time.perf_counter()-started})
        selected, audit = select_operations(scored, cfg['construction']['keep_fraction'], cfg['construction']['maximum_operations'])
        if len(selected) >= cfg['construction']['minimum_selected']:
            break
    if len(selected) < cfg['construction']['minimum_selected']:
        raise ValueError('Insufficient genuine selected operations; increase independent rollouts in a new run')
    if not any(abs(op['score']) > 1e-8 for op in scored):
        raise ValueError('Hindsight scores are all zero; refusing arbitrary tied SFT supervision')
    write_jsonl(output / 'scored.jsonl.gz', scored)
    write_jsonl(output / 'selected.jsonl', selected)
    write_json(output / 'selection.json', audit)
    write_json(output / 'COMPLETE.json', {'selected': len(selected), 'sampled': len(scored), 'replicas': replica + 1})
    return {'selected': len(selected), 'sampled': len(scored)}
