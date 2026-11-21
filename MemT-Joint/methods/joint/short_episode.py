"""Empty-state short episodes with batched, isolated on-policy memory branches."""
import copy

from methods.joint.batching import formation_many, evolution_many, continuation_many
from methods.joint.retrieval import sample_many
from methods.joint.rollout import blocks, new_state


def local_questions(questions, state, block, quota, rng):
    current = {t['turn_id'] for t in block.turns}
    past = set(state.sources) - current
    fresh = [q for q in questions if set(q['evidence']) & current]
    retained = [q for q in questions if set(q['evidence']) <= past]
    if not fresh or not retained:
        return []
    selected = rng.sample(fresh, min(len(fresh), (quota + 1) // 2))
    selected += rng.sample(retained, min(len(retained), quota // 2))
    remaining = [q for q in questions if q not in selected]
    return selected + rng.sample(remaining, min(len(remaining), quota - len(selected)))


def validate_short_weights(info, weights, skipped):
    allowed = {'U.local': ('no_qa_coverage', 'no_legal_candidates'), 'U.global': ('no_legal_candidates',)}
    if any(key not in allowed or reason not in allowed[key] for key, reason in skipped.items()):
        raise ValueError('Unsupported skipped training branch')
    for role, branches in weights.items():
        expected = 0
        for branch, mass in branches.items():
            mass = 0 if role + '.' + branch in skipped else mass
            expected += mass
            if abs(info['branch_weights'][role].get(branch, 0) - mass) > 1e-6:
                raise ValueError('Missing or changed branch weight without a matching skip record')
        if abs(info['role_weights'][role] - expected) > 1e-6 or bool(info['rows_by_role'][role]) != bool(expected):
            raise ValueError('Training samples or role weight disagree with skipped branches')


def collect_short(episode, questions, runtime, iteration):
    cfg = runtime.config['joint']
    sequence = list(blocks(episode, runtime.config['memory']['summary_context_turns']))
    if not 2 <= len(sequence) <= runtime.config['short_episode']['blocks']:
        raise ValueError('Short episode must contain 2..8 blocks')
    endpoint = len(episode['sessions']) - 1
    if any(q.get('eval_session') != endpoint or q.get('reviewed') is not True for q in questions):
        raise ValueError('Short-episode QA must be reviewed at the visible endpoint')
    if len(questions) < max(cfg['global_questions'], cfg['local_questions']):
        raise ValueError('Not enough distinct endpoint QA; never duplicate annotations')
    prefix = '%s/iteration-%d/short' % (episode['sample_id'], iteration)
    global_qs = runtime.random(prefix + '/questions').sample(questions, cfg['global_questions'])
    paths = [{'rows': [], 'snapshots': [], 'edits': {}, 'evaluations': [], 'operations': [], 'blocks': []}
             for _ in range(cfg['global_rollouts'])]
    states = [new_state(episode['sample_id'], runtime) for _ in paths]
    anchors, repair_anchors, coverage = [], [], []
    for bi, block in enumerate(sequence):
        for state in states:
            state.observe(block.session, block.turns, block.position, runtime.count)
        events = [prefix + '/global-%d/block-%d' % (gi, bi) for gi in range(len(paths))]
        formed = formation_many(states, block, runtime, [e + '/B' for e in events])
        evolved = evolution_many([f[0] for f in formed], [f[1] for f in formed], block, runtime,
                                 [e + '/U' for e in events])
        qs = local_questions(questions, states[0], block, cfg['local_questions'], runtime.random(prefix + '/anchor-qa-%d' % bi))
        current = {t['turn_id'] for t in block.turns}
        past = set(states[0].sources) - current
        coverage.append({'block': bi, 'write_qa': sum(bool(set(q['evidence']) & current) for q in questions),
                         'retention_qa': sum(set(q['evidence']) <= past for q in questions),
                         'pending_by_global_path': [len(f[1]) for f in formed]})
        if len(qs) == cfg['local_questions']:
            # Preserve the original path when possible; another sampled path
            # is equally on-policy and must not be ignored after a format error.
            for gi, f in enumerate(formed):
                if f[1]:
                    anchors.append((states[gi].clone(), f[0].clone(), copy.deepcopy(f[1]), block, bi, qs, gi))
                    break
            else:
                repair_anchors.append((states[0].clone(), block, bi, qs))
        if bi == len(sequence) - 1:
            # B can still be trained at the reviewed endpoint when Local-U is unavailable.
            b_qs = runtime.random(prefix + '/terminal-B-questions').sample(questions, cfg['local_questions'])
            terminal_b = (states[0].clone(), formed[0][0].clone(), copy.deepcopy(formed[0][1]), block, bi, b_qs, 0)
        for path, f, e in zip(paths, formed, evolved):
            path['rows'].extend(f[2] + e[1]); path['operations'].extend(f[3] + e[2])
            path['snapshots'].append({'position': list(block.position), 'cost': e[0].cost(runtime.count), 'state': e[0].to_dict()})
            path['edits'][tuple(block.position)] = e[3]
            path['blocks'].append(block)
        states = [e[0] for e in evolved]
    repair_calls, skipped = 0, {}
    if not any(any(c['pending_by_global_path']) for c in coverage):
        skipped['U.global'] = 'no_legal_candidates'
    if not anchors:
        if not repair_anchors:
            skipped['U.local'] = 'no_qa_coverage'
        else:
            initial, block, bi, qs = runtime.random(prefix + '/anchor-repair').choice(repair_anchors)
            events = [prefix + '/anchor-repair/candidate-%d/B' % i for i in range(cfg['builder_rollouts'])]
            repaired = formation_many([initial] * len(events), block, runtime, events)
            repair_calls = len(events)
            for f in repaired:
                if f[1]:
                    anchors.append((initial, f[0].clone(), copy.deepcopy(f[1]), block, bi, qs, None))
                    break
            else:
                skipped['U.local'] = 'no_legal_candidates'
    initial, after_b, pending, block, bi, qs, anchor_path = (
        runtime.random(prefix + '/anchor').choice(anchors) if anchors else terminal_b)
    requests = [(gi, state, q, prefix + '/global-%d/qa-%d' % (gi, qi))
                for gi, state in enumerate(states) for qi, q in enumerate(global_qs)]
    records = sample_many([r[1] for r in requests], [r[2] for r in requests], runtime,
                          [r[3] for r in requests], forest=False)
    for request, record in zip(requests, records):
        record['eval_position'] = list(sequence[-1].position)
        paths[request[0]]['evaluations'].append(record)
    b_events = [prefix + '/local-B/candidate-%d' % i for i in range(cfg['builder_rollouts'])]
    u_events = [] if 'U.local' in skipped else [prefix + '/local-U/candidate-%d' % i for i in range(cfg['updater_rollouts'])]
    b_formed = formation_many([initial] * len(b_events), block, runtime, [e + '/B' for e in b_events])
    b_evolved = evolution_many([f[0] for f in b_formed], [f[1] for f in b_formed], block, runtime,
                              [e + '/eval-U' for e in b_events], greedy=True)
    u_evolved = evolution_many([after_b] * len(u_events), [pending] * len(u_events), block, runtime,
                              [e + '/U' for e in u_events])
    local_states = [e[0] for e in b_evolved + u_evolved] + ([after_b] if u_events else [])
    events = b_events + u_events + ([prefix + '/before-U'] if u_events else [])
    local_states = continuation_many(local_states, sequence[bi+1:], runtime, events)
    requests = [(si, state, q, event + '/eval-R/qa-%d' % qi)
                for si, (state, event) in enumerate(zip(local_states, events)) for qi, q in enumerate(qs)]
    records = sample_many([r[1] for r in requests], [r[2] for r in requests], runtime,
                          [r[3] for r in requests], forest=False, greedy=True)
    evaluated = [[] for _ in local_states]
    for request, record in zip(requests, records):
        evaluated[request[0]].append(record)
    local_b, local_u = [], []
    for ci, (state, evolved) in enumerate(zip(local_states, b_evolved + u_evolved)):
        is_b = ci < len(b_events)
        rows, ops = (b_formed[ci][2], b_formed[ci][3]) if is_b else (evolved[1], evolved[2])
        item = {'rows': rows, 'operations': ops, 'evaluations': evaluated[ci],
                'before': [] if is_b else evaluated[-1], 'memory_cost': state.cost(runtime.count),
                'before_cost': (local_states[-1] if u_events else after_b).cost(runtime.count), 'edit_cost': evolved[3],
                'position': list(block.position), 'role': 'B' if is_b else 'U',
                'evaluation_horizon': 'short_episode_end',
                'anchor_hash': (initial if is_b else after_b).fingerprint()}
        (local_b if is_b else local_u).append(item)
    r_qs = runtime.random(prefix + '/r-questions').sample(questions, min(len(questions), cfg['retrieval_questions']))
    local_r = sample_many([states[i % len(states)] for i in range(len(r_qs))], r_qs, runtime,
                          [prefix + '/local-R/qa-%d' % i for i in range(len(r_qs))], forest=True)
    return {'episode': episode['sample_id'], 'iteration': iteration, 'global': paths,
            'local_B': [local_b], 'local_U': [local_u] if local_u else [], 'local_R': local_r,
            'eligible_questions': len(questions), 'skipped_local_anchors': len(sequence) - len(anchors),
            'short_episode': {'blocks': len(sequence), 'initial_state': 'empty', 'anchor_block': bi,
                              'remaining_blocks': len(sequence) - bi - 1, 'qa_count': len(questions),
                              'anchor_global_path': anchor_path, 'anchor_builder_resampled': anchor_path is None,
                              'anchor_builder_resample_calls': repair_calls, 'anchor_coverage': coverage,
                              'skipped_branches': skipped}}
