from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from memory.branch import InvalidAction
from memory.protocol import flatten, parse_calls, validate_call
from methods.joint.prompts import retrieval_prompt, retriever


SEARCH = {'search_summary': 'summary', 'search_personas': 'personas', 'search_facts': 'facts',
          'search_experiences': 'experiences', 'search_turns': 'turns'}


class MemoryOnlyView:
    """The diagnostic hides raw sources without changing the main tool path."""
    def __init__(self, db):
        self.db = db

    @staticmethod
    def clean(rows, nested=False):
        from memory.formatting import parse_memory_content
        rows = copy.deepcopy(rows)
        documents = rows['documents'] if nested else [rows['documents']]
        metadatas = rows['metadatas'] if nested else [rows['metadatas']]
        for docs, metas in zip(documents, metadatas):
            for i, meta in enumerate(metas):
                meta.pop('original_turns', None)
                docs[i] = meta.get('memory_content', parse_memory_content(docs[i])[0])
        return rows

    def get(self, *args, **kwargs):
        return self.clean(self.db.get(*args, **kwargs))

    def search(self, *args, **kwargs):
        return self.clean(self.db.search(*args, **kwargs), nested=True)


@dataclass
class Node:
    id: int
    parent: int
    depth: int
    messages: list
    rows: list = field(default_factory=list)
    observed: dict = field(default_factory=dict)
    displayed: list = field(default_factory=list)
    submitted: list = field(default_factory=list)
    children: list = field(default_factory=list)
    text: str = ''
    observation: str = ''
    valid: bool = True
    terminal: bool = False
    cost_tokens: int = 0
    ticket: str = ''
    answer: str = ''
    performance: float = 0.0
    evidence: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    stop_reason: str = ''
    fallback: bool = False
    answer_refs: list = field(default_factory=list)
    answer_performance: float = 0.0
    answer_evidence: float = None
    seen_turns: list = field(default_factory=list)


def prompt(question, mode, top_k=5, max_steps=6):
    return retrieval_prompt(question, mode, top_k, max_steps)


def step(state, node, text, runtime):
    calls = parse_calls(text)
    if len(calls) != 1:
        raise InvalidAction('Invalid tool-call format.')
    name, args = calls[0]['name'], calls[0]['arguments']
    db = MemoryOnlyView(state.journal) if runtime.mode == 'memory_only' else state.journal
    reader = retriever(db, runtime.mode, runtime.config['memory']['retrieval_topk'],
                       runtime.config['retrieval']['max_depth'], runtime.config.get('benchmark', 'locomo'))
    tool = reader.tool_map.get(name)
    if tool is None or not validate_call(calls[0], tool):
        raise InvalidAction('Invalid tool name or arguments.')
    if name == 'finish':
        answer = args.get('answer')
        if not answer.strip():
            raise InvalidAction('')
        node.answer = answer
        node.answer_refs = list(node.observed)
        node.terminal = True
        node.stop_reason = 'finish'
        return
    if name not in SEARCH or runtime.mode == 'memory_only' and name == 'search_turns':
        raise InvalidAction('Unknown or disabled search tool')
    seen = set(node.seen_turns)
    observation, metadata = tool(**args, sample_id=state.sample_id, seen_turns=seen)
    observation = observation or 'No matching memories.'
    if runtime.count(observation) > runtime.config['retrieval']['max_observation_tokens']:
        raise ValueError('Tool observation exceeds explicit observation budget')
    node.observation = observation
    node.cost_tokens += runtime.count(observation)
    node.seen_turns = sorted(seen)
    # Returned metadata alone does not prove its raw sources were displayed.
    for meta in metadata:
        ref = meta['memory_ref']
        sources = {}
        if runtime.mode == 'source_backed':
            for source in flatten(meta.get('source_turn_ids', [])):
                if source not in state.sources:
                    raise RuntimeError('Memory points to an unobserved source')
                if any(state.sources[source] in rendered for rendered in seen):
                    sources[source] = state.sources[source]
        node.observed[ref] = {'memory_ref': ref, 'sources': sources}
    node.displayed = sorted({s for card in node.observed.values() for s in card['sources']})


def sample_many(states, questions, runtime, events, forest=True, greedy=False):
    if not (len(states) == len(questions) == len(events)):
        raise ValueError('State/question/event count mismatch')
    config = runtime.config['retrieval']
    before = [state.fingerprint() for state in states]
    forests = [[[Node(0, -1, 0, prompt(q, runtime.mode, runtime.config['memory']['retrieval_topk'], config['max_depth']))] for _ in range(config['trees'] if forest else 1)]
               for q in questions]

    def extend(jobs):
        while jobs:
            batches, identities, active = [], [], []
            for qi, ti, pivot in jobs:
                parent = forests[qi][ti][pivot]
                messages = copy.deepcopy(parent.messages)
                if parent.id:
                    messages += [{'role': 'assistant', 'content': parent.text},
                                 {'role': 'user', 'content': 'The retrieved memories are: <observation>' + parent.observation +
                                  '</observation>\n Verify if you are confident enough to answer the <question>' +
                                  questions[qi]['question'] + '</question> correctly and sufficiently based on the above memories; '
                                  'if not, continue using the tools to retrieve more information.'}]
                if not runtime.prompt_fits(messages):
                    raise ValueError('Prompt exceeds explicit context budget')
                active.append((qi, ti, pivot))
                batches.append(messages)
                identities.append('%s/tree-%d/node-%d' % (events[qi], ti, len(forests[qi][ti])))
            generated = runtime.call_batch('R', batches, identities, greedy)
            following = []
            for (qi, ti, pivot), messages, (completion, row) in zip(active, batches, generated):
                nodes = forests[qi][ti]
                parent = nodes[pivot]
                node = Node(len(nodes), parent.id, parent.depth + 1, messages,
                            rows=[row], observed=copy.deepcopy(parent.observed), displayed=list(parent.displayed),
                            seen_turns=list(parent.seen_turns),
                            text=completion.text, cost_tokens=parent.cost_tokens + len(completion.response_ids))
                try:
                    step(states[qi], node, completion.text, runtime)
                except InvalidAction as error:
                    node.valid, node.terminal = False, True
                    node.stop_reason = str(error)
                    node.observation = str(error)
                nodes.append(node)
                parent.children.append(node.id)
                if not node.terminal and node.depth < config['max_depth']:
                    following.append((qi, ti, node.id))
            jobs = following

    extend([(qi, ti, 0) for qi, trees in enumerate(forests) for ti in range(len(trees))])
    if forest:
        rngs = [[random.Random(runtime.seed(event + '/pivots/' + str(ti))) for ti in range(config['trees'])]
                for event in events]
        for _ in range(config['expansion_rounds']):
            # Duplicate pivots are expanded in separate waves, preserving unique node IDs.
            choices = {}
            for qi, trees in enumerate(forests):
                for ti, nodes in enumerate(trees):
                    eligible = [n.id for n in nodes if not n.terminal and n.depth < config['max_depth']]
                    choices[(qi, ti)] = rngs[qi][ti].choices(eligible, k=config['pivots'])
            for pi in range(config['pivots']):
                extend([(qi, ti, pivots[pi]) for (qi, ti), pivots in choices.items()])
    for state, fingerprint in zip(states, before):
        if fingerprint != state.fingerprint():
            raise RuntimeError('Retriever mutated persistent memory')
    if runtime.config['evaluation'].get('fallback_answer', False):
        for q, trees in zip(questions, forests):
            for nodes in trees:
                for node in nodes[1:]:
                    if node.children or node.answer:
                        continue
                    node.fallback = True
                    path, cursor = [], node
                    while cursor.id:
                        path.append(cursor)
                        cursor = nodes[cursor.parent]
                    memories = '\n\n'.join(n.observation for n in reversed(path) if n.observation)
                    node.answer_refs = list(node.observed)
                    node.answer = runtime.fallback_answer(q, memories, node.rows[0]['event'])
    return [{'question': copy.deepcopy(q), 'trees': trees, 'state_hash': fingerprint, 'event': event}
            for q, trees, fingerprint, event in zip(questions, forests, before, events)]


def sample(state, question, runtime, event, forest=True, greedy=False):
    return sample_many([state], [question], runtime, [event], forest, greedy)[0]
