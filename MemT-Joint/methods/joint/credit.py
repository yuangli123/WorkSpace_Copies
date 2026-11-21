from __future__ import annotations

import math

from methods.retrieval.tree import answer_f1


def standardize(values):
    if not values:
        return []
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return [(v - mean) / (std + 1e-6) for v in values]


def evidence(question, sources):
    groups = question.get('evidence_groups') or ([question['evidence']] if question.get('evidence') else [])
    if not groups or question.get('invalid_evidence'):
        return None
    return max(len(set(sources) & set(g)) / len(set(g)) for g in groups if g)


def score_record(record, runtime, mot=False):
    cfg, question = runtime.config['reward'], record['question']
    for tree in record['trees']:
        for node in reversed(tree[1:]):
            answer_sources = {s for ref in node.answer_refs for s in node.observed[ref]['sources']}
            node.answer_evidence = evidence(question, answer_sources) if runtime.mode == 'source_backed' and node.answer else None
            node.answer_performance = answer_f1(node.answer, question['answer']) if node.answer else 0.0
            # finish(answer) no longer submits memory_refs.  For a valid direct
            # answer, score the evidence actually observed on that trajectory;
            # legacy submitted refs remain a compatibility fallback.
            evidence_refs = node.answer_refs or node.submitted
            sources = ({s for ref in evidence_refs for s in node.observed[ref]['sources']}
                       if node.terminal and node.valid else set(node.displayed))
            ev = evidence(question, sources) if runtime.mode == 'source_backed' else None
            node.evidence = ev
            if node.children:
                node.performance = sum(tree[c].performance for c in node.children) / len(node.children)
            else:
                # Fallback cannot repair an invalid R action's training reward.
                node.performance = node.answer_performance if node.valid and not node.fallback and node.stop_reason == 'finish' else 0.0
            node.reward = (node.performance + cfg['evidence'] * (ev or 0.0)
                           - cfg['retrieval_cost'] * node.cost_tokens / runtime.config['retrieval']['token_budget']) if node.valid else cfg['invalid']
    if mot:
        nodes = [n for t in record['trees'] for n in t[1:]]
        inter = standardize([n.reward for n in nodes])
        for tree in record['trees']:
            intra = standardize([n.reward for n in tree[1:]])
            for n, a in zip(tree[1:], intra):
                n.advantage = a
        for n, a in zip(nodes, inter):
            n.advantage += a
    # B/U task utility uses the same answer/fallback protocol as validation.
    leaf = record['trees'][0][-1]
    if not leaf.answer:
        return -cfg['retrieval_cost'] * min(1, leaf.cost_tokens / runtime.config['retrieval']['token_budget'])
    return leaf.answer_performance + cfg['evidence'] * (leaf.answer_evidence or 0.0) - cfg['retrieval_cost'] * min(1, leaf.cost_tokens / runtime.config['retrieval']['token_budget'])


def average_evaluations(records, runtime, balance_at=None):
    values = [score_record(r, runtime) for r in records]
    if balance_at is not None:
        buckets = [[], []]
        for record, value in zip(records, values):
            when = record['question'].get('eval_session')
            buckets[int(when is None or when >= balance_at)].append(value)
        available = [sum(b) / len(b) for b in buckets if b]
        return sum(available) / len(available) if available else None
    return sum(values) / len(values) if values else None



def construction_reward(rows, quality, runtime):
    invalid = runtime.config['reward']['invalid']
    if runtime.config.get('window') and rows:
        # A single failed invocation must not erase every other U action's signal.
        return sum(quality if r['valid'] else invalid for r in rows) / len(rows)
    return invalid if any(not r['valid'] for r in rows) else quality


def score_round(round_data, runtime):
    cfg = runtime.config['reward']
    prefix = '%s/iteration-%d/' % (round_data['episode'], round_data['iteration'])
    groups = {}
    for gi, path in enumerate(round_data['global']):
        for record in path['evaluations']:
            record['utility'] = score_record(record, runtime)
        by_block = {}
        for row in path['rows']:
            key = row['role'], tuple(row['position'])
            by_block.setdefault(key, []).append(row)
        for (role, position), rows in by_block.items():
            future = [r['utility'] for r in path['evaluations'] if tuple(r['eval_position']) >= position]
            snapshots = [s['cost'] for s in path['snapshots'] if tuple(s['position']) >= position]
            if not future:
                continue
            reward = sum(future) / len(future) - cfg['memory_cost'] * sum(snapshots) / len(snapshots)
            if role == 'U':
                reward -= cfg['edit_cost'] * path['edits'][position]
            reward = construction_reward(rows, reward, runtime)
            key = prefix + 'global/%s/%s' % (role, position)
            groups.setdefault(key, []).append({'rows': rows, 'reward': reward, 'role': role,
                                               'branch': 'global', 'candidate': gi})
    for role in ('B', 'U'):
        for ai, group in enumerate(round_data['local_' + role]):
            values = []
            for ci, item in enumerate(group):
                quality = average_evaluations(item['evaluations'], runtime, item['position'][0] if role == 'U' else None)
                reward = quality - cfg['memory_cost'] * item['memory_cost']
                if role == 'U':
                    before = average_evaluations(item['before'], runtime, item['position'][0]) - cfg['memory_cost'] * item['before_cost']
                    reward -= before + cfg['edit_cost'] * item['edit_cost']
                reward = construction_reward(item['rows'], reward, runtime)
                item['reward'] = reward
                if item['rows']:
                    values.append({'rows': item['rows'], 'reward': reward, 'role': role,
                                   'branch': 'local', 'candidate': ci, 'anchor_hash': item['anchor_hash']})
            groups[prefix + 'local/%s/%d' % (role, ai)] = values
    for record in round_data['local_R']:
        score_record(record, runtime, mot=True)
    rows = []
    group_counts = {}
    for group in groups.values():
        if group:
            key = group[0]['role'], group[0]['branch']
            group_counts[key] = group_counts.get(key, 0) + 1
    enabled = runtime.config['experiment']['train_roles']
    weights = runtime.config['joint']['branch_weights']
    for name, group in groups.items():
        for item, advantage in zip(group, standardize([g['reward'] for g in group])):
            role, branch = item['role'], item['branch']
            branch_weight = weights[role][branch]
            for row in item['rows']:
                row.update(group=name, branch=branch, candidate=item['candidate'], reward=item['reward'], advantage=advantage,
                           anchor_hash=item.get('anchor_hash'),
                           weight=branch_weight / (len(group) * group_counts[(role, branch)] * len(item['rows'])))
                if role in enabled and branch_weight:
                    row['training_eligible'] = True
                    rows.append(row)
    for qi, record in enumerate(round_data['local_R']):
        for ti, tree in enumerate(record['trees']):
            for node in tree[1:]:
                row = node.rows[0]
                row.update(group=record['event'], branch='mot', tree=ti, node=node.id, reward=node.reward,
                           anchor_hash=record['state_hash'],
                           training_eligible=True,
                           advantage=node.advantage, weight=weights['R']['mot'] /
                           (len(round_data['local_R']) * len(record['trees']) * (len(tree) - 1)))
                rows.append(row)
    return rows, groups
