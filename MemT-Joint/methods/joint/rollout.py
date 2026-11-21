from __future__ import annotations

import copy
from dataclasses import dataclass

from memory.branch import State, InvalidAction, builder_prompt, builder, updater_prompt, updater
from methods.joint.retrieval import sample_many


@dataclass
class Block:
    session: dict
    turns: list
    position: tuple


def new_state(sample_id, runtime):
    state = State(sample_id, runtime.ranker)
    if runtime.config['backend'] != 'fixture' and runtime.config['memory'].get('backend', 'chroma') == 'chroma':
        state.enable_chroma()
    return state


def blocks(episode, width, horizon=None):
    for si, session in enumerate(episode['sessions'][:horizon]):
        for offset in range(0, len(session['turns']), width):
            yield Block(session, session['turns'][offset:offset + width], (si, offset))


def formation(state, block, runtime, event, greedy=False):
    completion, row = runtime.call('B', builder_prompt(state, block.turns), event, greedy=greedy)
    try:
        updated, candidates, operations = builder(state, block.turns, completion, event)
        valid = all(op['valid'] for op in operations)
    except InvalidAction as error:
        updated, candidates, operations, valid = state.clone(), [], [], False
        row['error'] = str(error)
    row.update(valid=valid, position=list(block.position), state_hash=state.fingerprint())
    return updated, candidates, [row], operations


def evolution(state, candidates, block, runtime, event, greedy=False):
    state = state.clone()
    rows, operations = [], []
    before = state.to_dict()
    for index, item in enumerate(candidates):
        messages, related = updater_prompt(state, item, runtime.config['memory']['update_retrieval_topk'])
        identity = event + '/item-' + str(index)
        completion, row = runtime.call('U', messages, identity, greedy=greedy)
        row.update(position=list(block.position), state_hash=state.fingerprint())
        try:
            state, atomic = updater(state, item, related, completion, identity)
            operations.extend(atomic)
            row['valid'] = all(op['valid'] for op in atomic)
        except InvalidAction as error:
            row.update(valid=False, error=str(error))
        rows.append(row)
    # Charge actual changed documents, not the size of an unchanged database.
    old = before['collections']
    changed = sum(runtime.count(doc) for collection, values in state.journal.db.collections.items()
                  for key, (doc, meta) in values.items()
                  if key not in old.get(collection, {}) or old[collection][key][0] != doc)
    input_size = sum(runtime.count(state.sources[t['turn_id']]) for t in block.turns)
    edit_cost = changed / max(1, changed + input_size)
    return state, rows, operations, edit_cost


def eligible(questions, session_index, total_sessions, terminal=False, session_complete=True):
    result = []
    for question in questions:
        when = question.get('eval_session')
        if when is None:
            if terminal and session_index == total_sessions - 1:
                result.append(question)
        elif when < session_index or when == session_index and session_complete:
            result.append(question)
    return result


def evaluate_state(state, questions, runtime, event):
    return sample_many([state] * len(questions), questions, runtime,
                       [event + '/qa-' + str(i) for i in range(len(questions))], forest=False, greedy=True)


def continue_state(state, remaining, runtime, event):
    state = state.clone()
    for index, block in enumerate(remaining):
        state.observe(block.session, block.turns, block.position, runtime.count)
        identity = event + '/continuation-%d' % index
        state, pending, _, _ = formation(state, block, runtime, identity + '/B', greedy=True)
        state, _, _, _ = evolution(state, pending, block, runtime, identity + '/U', greedy=True)
    return state


def collect(episode, questions, runtime, iteration, horizon=None):
    """Collect all branches under one behavior policy; scoring is deferred."""
    cfg = runtime.config['joint']
    prefix = '%s/iteration-%d' % (episode['sample_id'], iteration)
    width = runtime.config['memory']['summary_context_turns']
    all_blocks = list(blocks(episode, width, horizon))
    if not all_blocks:
        raise ValueError('Empty episode')
    last_si = all_blocks[-1].position[0]
    full = last_si == len(episode['sessions']) - 1
    terminal_questions = eligible(questions, last_si, len(episode['sessions']), terminal=full)
    if not terminal_questions:
        raise ValueError('No eligible QA at this horizon; add reviewed training probes or increase the horizon')
    global_questions = runtime.random(prefix + '/global-questions').sample(
        terminal_questions, min(cfg['global_questions'], len(terminal_questions)))
    paths, anchors, global_requests = [], [], []
    for gi in range(cfg['global_rollouts']):
        state = new_state(episode['sample_id'], runtime)
        path = {'rows': [], 'snapshots': [], 'edits': {}, 'evaluations': [], 'operations': [], 'blocks': []}
        for bi, block in enumerate(all_blocks):
            state.observe(block.session, block.turns, block.position, runtime.count)
            initial = state.clone()
            event = prefix + '/global-%d/block-%d' % (gi, bi)
            after_b, pending, b_rows, b_ops = formation(state, block, runtime, event + '/B')
            after_u, u_rows, u_ops, edit = evolution(after_b, pending, block, runtime, event + '/U')
            path['rows'].extend(b_rows + u_rows)
            path['operations'].extend(b_ops + u_ops)
            path['snapshots'].append({'position': list(block.position), 'cost': after_u.cost(runtime.count),
                                      'state': after_u.to_dict()})
            path['edits'][tuple(block.position)] = edit
            path['blocks'].append(block)
            if gi == 0:
                anchors.append((initial, after_b.clone(), copy.deepcopy(pending), block, bi))
            state = after_u
        for qi, q in enumerate(global_questions):
            when = q.get('eval_session', last_si)
            when = last_si if when is None else when
            snapshot = next(s for s in reversed(path['snapshots']) if s['position'][0] == when)
            query_state = State.from_dict(snapshot['state'], runtime.ranker)
            global_requests.append((query_state, q, prefix + '/global-%d/qa-%d' % (gi, qi), gi, snapshot['position']))
        paths.append(path)
    records = sample_many([r[0] for r in global_requests], [r[1] for r in global_requests], runtime,
                          [r[2] for r in global_requests], forest=False)
    for request, record in zip(global_requests, records):
        record['eval_position'] = request[4]
        paths[request[3]]['evaluations'].append(record)
    local_b, local_u = [], []
    candidates = []
    for anchor in anchors:
        block, bi = anchor[3], anchor[4]
        qs = eligible(questions, block.position[0], len(episode['sessions']), terminal=full and bi == len(all_blocks)-1,
                      session_complete=block.position[1] + len(block.turns) == len(block.session['turns']))
        delayed = False
        if not qs and full and cfg['local_evaluation'] == 'current_or_continuation':
            qs = [q for q in questions if q.get('eval_session') is None]
            delayed = bool(qs)
        if qs:
            candidates.append((anchor, qs, delayed))
    # Anchors are sampled independently from rewards, before candidate scoring.
    rng = runtime.random(prefix + '/anchor')
    chosen = rng.sample(candidates, min(cfg['local_anchors'], len(candidates)))
    for ai, ((initial, after_b, pending, block, bi), qs, delayed) in enumerate(chosen):
        past = [q for q in qs if q.get('eval_session') is not None and q['eval_session'] < block.position[0]]
        current = [q for q in qs if q not in past]
        quota = cfg['local_questions']
        local_questions = current[:(quota + 1) // 2] + past[:quota // 2]
        local_questions += [q for q in qs if q not in local_questions][:quota - len(local_questions)]
        for role, output, count in (('B', local_b, cfg['builder_rollouts']), ('U', local_u, cfg['updater_rollouts'])):
            if not count or role == 'U' and not pending:
                continue
            baseline_state = continue_state(after_b, all_blocks[bi+1:], runtime, prefix + '/anchor-%d/before-U' % ai) if delayed and role == 'U' else after_b
            baseline = evaluate_state(baseline_state, local_questions, runtime, prefix + '/anchor-%d/before-U' % ai) if role == 'U' else []
            group = []
            for ci in range(count):
                event = prefix + '/local-%s/anchor-%d/candidate-%d' % (role, ai, ci)
                if role == 'B':
                    state, items, rows, ops = formation(initial, block, runtime, event + '/B')
                    state, _, _, edit = evolution(state, items, block, runtime, event + '/eval-U', greedy=True)
                else:
                    state, rows, ops, edit = evolution(after_b, pending, block, runtime, event + '/U')
                if delayed:
                    state = continue_state(state, all_blocks[bi+1:], runtime, event)
                evaluations = evaluate_state(state, local_questions, runtime, event + '/eval-R')
                group.append({'rows': rows, 'operations': ops, 'evaluations': evaluations, 'before': baseline,
                              'memory_cost': state.cost(runtime.count), 'before_cost': baseline_state.cost(runtime.count),
                              'edit_cost': edit, 'position': list(block.position), 'role': role,
                              'evaluation_horizon': 'terminal_continuation' if delayed else 'current',
                              'anchor_hash': (initial if role == 'B' else after_b).fingerprint()})
            output.append(group)
    # Use immutable final states for R; each forest has exactly one state-query root.
    shuffled = list(terminal_questions)
    rng.shuffle(shuffled)
    local_r, r_states, r_questions, r_events = [], [], [], []
    if 'R' in runtime.config['experiment']['train_roles']:
        for qi, q in enumerate(shuffled[:cfg['retrieval_questions']]):
            snapshots = paths[qi % len(paths)]['snapshots']
            when = q.get('eval_session')
            snapshot = snapshots[-1] if when is None else next(s for s in reversed(snapshots) if s['position'][0] == when)
            state = State.from_dict(snapshot['state'], runtime.ranker)
            r_states.append(state)
            r_questions.append(q)
            r_events.append(prefix + '/local-R/qa-%d' % qi)
        local_r = sample_many(r_states, r_questions, runtime, r_events, forest=True)
    return {'episode': episode['sample_id'], 'iteration': iteration, 'global': paths,
            'local_B': local_b, 'local_U': local_u, 'local_R': local_r,
            'skipped_local_anchors': len(anchors) - len(candidates), 'eligible_questions': len(terminal_questions)}
