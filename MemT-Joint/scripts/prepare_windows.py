"""Offline train-only QA review and frozen-base prefix snapshots; never an RL step."""
import argparse
import copy
import json
import time
from pathlib import Path

from experiments.io import read_json, read_jsonl, write_json, write_jsonl, sha256
from experiments.joint_config import load
from memory.branch import State, Ranker, digest
from methods.joint.rollout import blocks, formation, evolution
from methods.retrieval.tree import answer_f1
from models.joint import Runtime
from models.vllm import VLLMPolicy


def parse(text):
    from experiments.window import parse_response
    return parse_response(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--reuse')
    parser.add_argument('--prefix-snapshots')
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.output)
    assert read_json(data / 'partition.json')['split'] == 'train'
    assert not out.exists(), 'Use a new offline preparation run'
    out.mkdir(parents=True)
    episodes = read_jsonl(data / 'episodes.jsonl')
    assert len(episodes) == 1
    episode = episodes[0]
    cfg = load('configs/common-a100x2-50.json')
    cfg['experiment']['train_roles'] = ['B', 'U', 'R']
    policy = VLLMPolicy(cfg['models']['base'], cfg)
    runtime = Runtime(policy, cfg, ranker=Ranker(cfg['memory']['embedding_model']))
    sequence = list(blocks(episode, cfg['memory']['summary_context_turns']))
    windows, accepted, rejected = [], [], []
    started = time.perf_counter()

    def request(name, instruction, value, seed):
        messages = [{'role': 'system', 'content': instruction + ' Return JSON only, no markdown or reasoning.'},
                    {'role': 'user', 'content': json.dumps(value, ensure_ascii=False)}]
        if args.reuse:
            cached = Path(args.reuse) / 'reviews' / (name + '.json')
            if cached.is_file():
                value = read_json(cached)
                if value['messages'] == messages:
                    file = out / 'reviews' / cached.name
                    value['parser'] = 'complete-json-with-trailing-closer-repair-v1'
                    write_json(file, value)
                    return parse(value['response']), str(file.relative_to(out)), sha256(file)
        result = policy.complete_batch([messages], [seed], max_new_tokens=4096,
                                       temperature=0, stop_at_tool=False)[0]
        file = out / 'reviews' / (name + '.json')
        write_json(file, {'messages': messages, 'response': result.text,
                          'prompt_tokens': len(result.prompt_ids), 'response_tokens': len(result.response_ids),
                          'checkpoint': cfg['models']['base'], 'offline_only': True,
                          'parser': 'complete-json-with-trailing-closer-repair-v1'})
        return parse(result.text), str(file.relative_to(out)), sha256(file)

    try:
        with policy.phase():
            for endpoint in (2, 8, 18):
                prefix = copy.deepcopy(episode)
                prefix['sessions'] = prefix['sessions'][:endpoint + 1]
                end = sum(b.position[0] <= endpoint for b in sequence)
                start = end - 8
                assert start >= 0
                visible = {t['turn_id']: t for s in prefix['sessions'] for t in s['turns']}
                recent = {t['turn_id'] for b in sequence[start:end] for t in b.turns}
                if endpoint == 2:
                    candidates = read_jsonl(args.candidates)
                else:
                    # Reuse historical questions authored before later dialogue was exposed.
                    seed_questions = read_jsonl(args.candidates)
                    candidates = [copy.deepcopy(seed_questions[i]) for i in (0, 1, 4, 5)]
                    target_ids = (['D9:2', 'D9:6', 'D9:12', 'D9:16'] if endpoint == 8 else
                                  ['D19:1', 'D19:2', 'D19:4', 'D19:5', 'D18:19', 'D18:21'])
                    window_context = [dict(session, turns=[t for t in session['turns'] if t['turn_id'] in recent])
                                      for session in prefix['sessions']]
                    window_context = [session for session in window_context if session['turns']]
                    for index, target in enumerate(target_ids):
                        assert target in recent
                        q, _, _ = request('s%d-target-%d' % (endpoint+1, index),
                            'Create ONE concrete factual short-answer question from the target turn. '
                            'You see only the current dialogue window, with no later dialogue. '
                            'Answer must be a short phrase of at most SIX words taken directly from the target turn. '
                            'Ask about a specific group, object, activity, place or milestone. '
                            'Do NOT ask when, opinions, What did X say, general encouragement, or cutoff date. '
                            'Identify the correct speaker. Avoid tautology and do not include the answer in the question. '
                            'Return one JSON object: question, answer, evidence (a list containing ONLY target_turn_id).',
                            {'window': window_context, 'target_turn_id': target,
                             'cutoff_time': prefix['sessions'][-1]['time']}, 7200+endpoint*100+index)
                        assert isinstance(q, dict) and isinstance(q.get('answer'), str)
                        source_session = next(session for session in prefix['sessions']
                                              if any(t['turn_id'] == target for t in session['turns']))
                        question = q['question'].strip()
                        q['question'] = 'As reported at %s, %s' % (source_session['time'], question[0].lower()+question[1:])
                        q['target_turn_id'], q['probe_type'] = target, 'window_fact'
                        candidates.append(q)
                assert isinstance(candidates, list)
                # This independent request has questions but NO proposed answers/evidence.
                blind, blind_file, blind_hash = request('s%d-blind' % (endpoint + 1),
                    'Independently answer each question using only the visible dialogue prefix. '
                    'Do not guess. Resolve relative dates against the source session date, not cutoff date. '
                    'Return a JSON array in question order with answer (short phrase), evidence (exact turn IDs), '
                    'answerable (boolean). If ambiguous or unsupported use answerable=false.',
                    {'prefix': prefix, 'questions': [q['question'] for q in candidates],
                     'cutoff_time': prefix['sessions'][-1]['time']}, 8200 + endpoint)
                assert len(blind) == len(candidates)
                review_files = []
                verdicts = []
                for index, (q, b) in enumerate(zip(candidates, blind)):
                    verdict, review_file, review_hash = request('s%d-audit-item%d' % (endpoint + 1, index),
                        'Audit this one proposed QA and the independently obtained answer against the visible prefix. '
                        'Check exact source support, speaker, chronology and question ambiguity. '
                        'accept=true only if both answers are semantically equivalent and independently supported, '
                        'all proposed evidence IDs exist, and question has a clear answer at cutoff. '
                        'Reject unsupported aggregations, incomplete evidence or ambiguous dates. '
                        'Return one JSON object: accept (boolean), reason (brief).',
                        {'prefix': prefix, 'cutoff_time': prefix['sessions'][-1]['time'],
                         'candidates': [{'question': q['question'], 'answer': q['answer'], 'evidence': q['evidence'],
                                         'blind': b}]}, 9200 + endpoint * 100 + index)
                    assert isinstance(verdict, dict) and isinstance(verdict.get('accept'), bool)
                    verdicts.append(verdict)
                    review_files.append((review_file, review_hash))
                for index, (q, b, verdict) in enumerate(zip(candidates, blind, verdicts)):
                    review_file, review_hash = review_files[index]
                    refs = set(q.get('evidence', []))
                    import re
                    normalize = lambda value: ' '.join(re.findall(r'\w+', str(value).lower()))
                    target = q.get('target_turn_id')
                    literal = (not target or (refs == {target} and normalize(q['answer']) in normalize(visible[target]['text'])))
                    valid = (literal and verdict.get('accept') is True and b.get('answerable') is True and refs
                             and refs <= visible.keys() and set(b.get('evidence', [])) <= visible.keys()
                             and bool(b.get('evidence')) and max(len(str(a).split()) for a in (q['answer'] if isinstance(q['answer'], list) else [q['answer']])) <= 8 and answer_f1(str(b.get('answer', '')), q['answer']) >= .5)
                    row = dict(q, sample_id=episode['sample_id'],
                        qa_id='%s:window-v2:s%d:%02d' % (episode['sample_id'], endpoint + 1, index),
                        eval_session=endpoint, question_time=prefix['sessions'][-1]['time'],
                        benchmark='locomo-window-probe', reviewed=bool(valid),
                        source_cutoff={'prefix_sha256': digest(prefix), 'episodes_sha256': sha256(data / 'episodes.jsonl')},
                        window={'start': start, 'end': end},
                        evidence_quotes=[visible[r] for r in q.get('evidence', []) if r in visible],
                        review={'blind_file': blind_file, 'blind_sha256': blind_hash,
                                'audit_file': review_file, 'audit_sha256': review_hash, 'index': index, 'audit_index': 0,
                                'blind_answer': b, 'verdict': verdict,
                                'independent_calls_same_checkpoint': True, 'human_reviewed': False})
                    (accepted if valid else rejected).append(row)
                current = [q for q in accepted if q['eval_session'] == endpoint]
                assert len(current) >= 4, 'Fewer than four independently checked QA at cutoff'
                assert sum(bool(set(q['evidence']) & recent) for q in current) >= 2
                windows.append({'sample_id': episode['sample_id'], 'eval_session': endpoint,
                                'start': start, 'end': end, 'snapshot': 'snapshots/block-%03d.json' % start})
                write_jsonl(out / 'reviewed-probes.jsonl', accepted)
                write_jsonl(out / 'rejected-probes.jsonl', rejected)
                write_json(out / 'PROGRESS.json', {'phase': 'offline_qa', 'completed_cutoffs': len(windows),
                                                 'accepted': len(accepted), 'elapsed': time.perf_counter()-started})
            if args.prefix_snapshots:
                import shutil
                previous_root = Path(args.prefix_snapshots).resolve()
                previous = read_json(previous_root / 'bundle.json')
                assert previous['episodes_sha256'] == sha256(data / 'episodes.jsonl')
                for w in windows:
                    prior = next(p for p in previous['windows'] if p['start'] == w['start'] and p['end'] == w['end'])
                    src = (previous_root / prior['snapshot']).resolve()
                    assert previous_root in src.parents and sha256(src) == prior['snapshot_sha256']
                    destination = out / w['snapshot']
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, destination)
            else:
                state = State(episode['sample_id'], runtime.ranker)
                starts = {w['start'] for w in windows}
                for bi, block in enumerate(sequence[:max(starts) + 1]):
                    if bi in starts:
                        write_json(out / ('snapshots/block-%03d.json' % bi), state.to_dict())
                    if bi == max(starts):
                        break
                    state.observe(block.session, block.turns, block.position, runtime.count)
                    event = 'offline-frozen-prefix/%d' % bi
                    state, pending, _, _ = formation(state, block, runtime, event + '/B', greedy=True)
                    state, _, _, _ = evolution(state, pending, block, runtime, event + '/U', greedy=True)
                    write_json(out / 'PROGRESS.json', {'phase': 'offline_prefix_construction',
                        'completed_blocks': bi + 1, 'total_blocks': max(starts),
                        'policy_calls': len(runtime.calls), 'elapsed': time.perf_counter()-started})
            for w in windows:
                w['snapshot_sha256'] = sha256(out / w['snapshot'])
            # Bind model files as well as source/QA/snapshot identities.
            model = Path(cfg['models']['base'])
            model_hashes = {str(p.relative_to(model)): sha256(p) for p in model.rglob('*')
                            if p.is_file() and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt', '.tiktoken')}
            if args.prefix_snapshots:
                assert previous['frozen_prefix_model'] == model_hashes
            bundle = {'version': 'memt-window-v2', 'split': 'train', 'windows': windows,
                      'probes': 'reviewed-probes.jsonl', 'probes_sha256': sha256(out / 'reviewed-probes.jsonl'),
                      'episodes_sha256': sha256(data / 'episodes.jsonl'), 'frozen_prefix_model': model_hashes,
                      'offline_seconds': time.perf_counter()-started,
                      'prefix_snapshot_source': str(Path(args.prefix_snapshots).resolve()) if args.prefix_snapshots else None,
                      'prefix_snapshot_source_bundle_sha256': sha256(Path(args.prefix_snapshots) / 'bundle.json') if args.prefix_snapshots else None,
                      'qa_review': 'independent stateless calls of same frozen base; assistant audit pending',
                      'assistant_audit_required': True}
            write_json(out / 'bundle.json', bundle)
            write_json(out / 'PREPARATION_COMPLETE.json', {'bundle_sha256': sha256(out / 'bundle.json'),
                                                        'formal_updates': 0})
    finally:
        policy.close()


if __name__ == '__main__':
    main()
