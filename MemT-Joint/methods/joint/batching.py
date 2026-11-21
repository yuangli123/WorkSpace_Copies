"""Batch independent memory branches while preserving per-branch update order."""
import copy

from memory.branch import InvalidAction, builder_prompt, builder, updater_prompt, updater


def formation_many(states, block, runtime, events, greedy=False):
    results = runtime.call_batch('B', [builder_prompt(s, block.turns) for s in states], events, greedy)
    outputs = []
    for state, event, (completion, row) in zip(states, events, results):
        try:
            updated, pending, operations = builder(state, block.turns, completion, event)
            row['valid'] = all(op['valid'] for op in operations)
        except InvalidAction as error:
            updated, pending, operations = state.clone(), [], []
            row.update(valid=False, error=str(error))
        row.update(position=list(block.position), state_hash=state.fingerprint())
        outputs.append((updated, pending, [row], operations))
    return outputs


def evolution_many(states, candidates, block, runtime, events, greedy=False):
    states = [s.clone() for s in states]
    before = [s.to_dict()['collections'] for s in states]
    rows, operations = [[] for _ in states], [[] for _ in states]
    # Dependence within one branch is sequential; independent branches share a batch.
    for index in range(max(map(len, candidates), default=0)):
        active = [i for i, pending in enumerate(candidates) if index < len(pending)]
        prompts = [updater_prompt(states[i], candidates[i][index], runtime.config['memory']['update_retrieval_topk'])
                   for i in active]
        identities = [events[i] + '/item-%d' % index for i in active]
        results = runtime.call_batch('U', [p[0] for p in prompts], identities, greedy)
        for i, (_, related), event, (completion, row) in zip(active, prompts, identities, results):
            row.update(position=list(block.position), state_hash=states[i].fingerprint())
            try:
                states[i], atomic = updater(states[i], candidates[i][index], related, completion, event)
                operations[i].extend(atomic)
                row['valid'] = all(op['valid'] for op in atomic)
            except InvalidAction as error:
                row.update(valid=False, error=str(error))
            rows[i].append(row)
    outputs = []
    for i, state in enumerate(states):
        changed = sum(runtime.count(doc) for collection, values in state.journal.db.collections.items()
                      for key, (doc, _) in values.items()
                      if key not in before[i].get(collection, {}) or before[i][collection][key][0] != doc)
        size = sum(runtime.count(state.sources[t['turn_id']]) for t in block.turns)
        outputs.append((state, rows[i], operations[i], changed / max(1, changed + size)))
    return outputs


def continuation_many(states, remaining, runtime, events):
    states = [s.clone() for s in states]
    for bi, block in enumerate(remaining):
        for state in states:
            state.observe(block.session, block.turns, block.position, runtime.count)
        identities = [e + '/continuation-%d' % bi for e in events]
        formed = formation_many(states, block, runtime, [e + '/B' for e in identities], greedy=True)
        evolved = evolution_many([f[0] for f in formed], [f[1] for f in formed], block, runtime,
                                 [e + '/U' for e in identities], greedy=True)
        states = [e[0] for e in evolved]
    return states
