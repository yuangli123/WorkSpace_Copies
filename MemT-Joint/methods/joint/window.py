"""Historical prefix-window runner; new short episodes always start empty."""
import copy

from memory.branch import State
from methods.joint.batching import formation_many, evolution_many, continuation_many
from methods.joint.retrieval import sample_many
from methods.joint.rollout import blocks


def window_questions(questions, recent_sources, quota, rng):
    new = [q for q in questions if set(q['evidence']) & recent_sources]
    old = [q for q in questions if not set(q['evidence']) & recent_sources]
    if len(new) < (quota+1)//2 or len(old) < quota//2:
        raise ValueError('Reviewed window QA must cover both current writes and prefix retention')
    return rng.sample(new, (quota+1)//2) + rng.sample(old, quota//2)


def collect_window(episode, questions, initial_state, spec, runtime, iteration):
    cfg = runtime.config['joint']
    prefix = '%s/iteration-%d/window-%d-%d' % (episode['sample_id'], iteration, spec['start'], spec['end'])
    all_blocks = list(blocks(episode, runtime.config['memory']['summary_context_turns']))[spec['start']:spec['end']]
    assert len(all_blocks) == 8 and all_blocks[-1].position[0] == spec['eval_session']
    terminal = [q for q in questions if q.get('eval_session') == spec['eval_session']]
    if len(terminal) < max(cfg['global_questions'], cfg['local_questions'], cfg['retrieval_questions']):
        raise ValueError('Not enough reviewed endpoint QA; never duplicate questions to meet quota')
    recent = {t['turn_id'] for b in all_blocks for t in b.turns}
    chosen_qs = window_questions(terminal, recent, cfg['global_questions'], runtime.random(prefix + '/questions'))
    paths = [{'rows': [], 'snapshots': [], 'edits': {}, 'evaluations': [], 'operations': [], 'blocks': []}
             for _ in range(cfg['global_rollouts'])]
    states = [initial_state.clone() for _ in paths]
    anchors = []
    for bi, block in enumerate(all_blocks):
        for state in states:
            state.observe(block.session, block.turns, block.position, runtime.count)
        before = states[0].clone()
        events = [prefix + '/global-%d/block-%d' % (gi, bi) for gi in range(len(paths))]
        formed = formation_many(states, block, runtime, [e + '/B' for e in events])
        evolved = evolution_many([f[0] for f in formed], [f[1] for f in formed], block, runtime,
                                 [e + '/U' for e in events])
        # Structural eligibility, selected before rewards; U must have an actual candidate.
        if formed[0][1]:
            anchors.append((before, formed[0][0].clone(), copy.deepcopy(formed[0][1]), block, bi))
        for path, f, e in zip(paths, formed, evolved):
            path['rows'].extend(f[2] + e[1])
            path['operations'].extend(f[3] + e[2])
            path['snapshots'].append({'position': list(block.position), 'cost': e[0].cost(runtime.count),
                                      'state': e[0].to_dict()})
            path['edits'][tuple(block.position)] = e[3]
            path['blocks'].append(block)
        states = [e[0] for e in evolved]
    requests = [(gi, state, q, prefix + '/global-%d/qa-%d' % (gi, qi))
                for gi, state in enumerate(states) for qi, q in enumerate(chosen_qs)]
    records = sample_many([r[1] for r in requests], [r[2] for r in requests], runtime,
                          [r[3] for r in requests], forest=False)
    for (gi, _, _, _), record in zip(requests, records):
        record['eval_position'] = list(all_blocks[-1].position)
        paths[gi]['evaluations'].append(record)
    if not anchors:
        raise ValueError('No formation candidates in window: Local U cannot be trained; inspect Builder output')
    initial, after_b, pending, block, bi = runtime.random(prefix + '/anchor').choice(anchors)
    local_qs = window_questions(terminal, recent, cfg['local_questions'], runtime.random(prefix + '/local-questions'))
    b_events = [prefix + '/local-B/candidate-%d' % ci for ci in range(cfg['builder_rollouts'])]
    u_events = [prefix + '/local-U/candidate-%d' % ci for ci in range(cfg['updater_rollouts'])]
    b_formed = formation_many([initial] * len(b_events), block, runtime, [e + '/B' for e in b_events])
    b_evolved = evolution_many([f[0] for f in b_formed], [f[1] for f in b_formed], block, runtime,
                              [e + '/eval-U' for e in b_events], greedy=True)
    u_evolved = evolution_many([after_b] * len(u_events), [pending] * len(u_events), block, runtime,
                              [e + '/U' for e in u_events])
    local_states = [e[0] for e in b_evolved + u_evolved] + [after_b]
    local_events = b_events + u_events + [prefix + '/before-U']
    local_states = continuation_many(local_states, all_blocks[bi+1:], runtime, local_events)
    queries = [(si, state, q, event + '/eval-R/qa-%d' % qi)
               for si, (state, event) in enumerate(zip(local_states, local_events)) for qi, q in enumerate(local_qs)]
    records = sample_many([q[1] for q in queries], [q[2] for q in queries], runtime,
                          [q[3] for q in queries], forest=False, greedy=True)
    evaluated = [[] for _ in local_states]
    for request, record in zip(queries, records):
        evaluated[request[0]].append(record)
    local_b, local_u = [], []
    for ci, (state, e) in enumerate(zip(local_states[:-1], b_evolved + u_evolved)):
        is_b = ci < len(b_events)
        rows, ops = (b_formed[ci][2], b_formed[ci][3]) if is_b else (e[1], e[2])
        item = {'rows': rows, 'operations': ops, 'evaluations': evaluated[ci],
                'before': [] if is_b else evaluated[-1], 'memory_cost': state.cost(runtime.count),
                'before_cost': local_states[-1].cost(runtime.count), 'edit_cost': e[3],
                'position': list(block.position), 'role': 'B' if is_b else 'U',
                'evaluation_horizon': 'window_end',
                'anchor_hash': (initial if is_b else after_b).fingerprint()}
        (local_b if is_b else local_u).append(item)
    r_qs = window_questions(terminal, recent, cfg['retrieval_questions'], runtime.random(prefix + '/r-questions'))
    local_r = sample_many([states[qi % len(states)] for qi in range(len(r_qs))], r_qs, runtime,
                          [prefix + '/local-R/qa-%d' % qi for qi in range(len(r_qs))], forest=True)
    return {'episode': episode['sample_id'], 'iteration': iteration, 'global': paths,
            'local_B': [local_b], 'local_U': [local_u], 'local_R': local_r,
            'skipped_local_anchors': len(all_blocks)-len(anchors), 'eligible_questions': len(terminal),
            'window': dict(spec), 'prefix_state_hash': initial_state.fingerprint(),
            'window_terminal_reward': True}
