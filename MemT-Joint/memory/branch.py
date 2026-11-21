"""Isolated four-tier branches; embeddings are shared, mutable memory is not."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import time

from memory.journal import JournalDB, ListDB
from memory.protocol import flatten, parse_calls, validate_call
from memory.formation import MemoryFormation, CreateRawTool
from memory.update import MemoryUpdate


class InvalidAction(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Ranker:
    def __init__(self, model=None, fixture=False):
        self.fixture, self.model_path = fixture, model
        self.encoder, self.cache = None, {}

    def vectors(self, texts):
        if self.encoder is None:
            from pathlib import Path
            from sentence_transformers import SentenceTransformer
            if not self.model_path or not Path(self.model_path).is_dir():
                raise ValueError('An existing local BGE model is required for real experiments')
            self.encoder = SentenceTransformer(self.model_path, device='cpu')
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            for text, vector in zip(missing, self.encoder.encode(missing, normalize_embeddings=True)):
                self.cache[text] = vector
        return [self.cache[t] for t in texts]

    def rank(self, query, documents):
        if self.fixture:
            words = set(re.findall(r'\w+', query.lower()))
            return sorted(range(len(documents)), key=lambda i: (-len(words & set(re.findall(r'\w+', documents[i].lower()))), i))
        vectors = self.vectors([query] + documents)
        return sorted(range(len(documents)), key=lambda i: (-float(vectors[0] @ vectors[i + 1]), i))


class BranchDB(ListDB):
    def __init__(self, ranker):
        super().__init__()
        self.ranker = ranker

    def search(self, name, query_texts, top_k=5, **kwargs):
        result = {'ids': [], 'documents': [], 'metadatas': []}
        rows = self.get(name)
        for query in query_texts:
            indices = self.ranker.rank(query, rows['documents'])[:top_k] if rows['ids'] else []
            for key in result:
                result[key].append([copy.deepcopy(rows[key][i]) for i in indices])
        return result


class ChromaBranchDB(BranchDB):
    """Isolated logical branches with an incrementally synchronized search view."""
    def __init__(self, ranker, collections):
        super().__init__(ranker)
        from memory.settings import VectorDBConfig
        from memory.vector_db import ChromaVectorDB
        # Chroma caches systems by path: a new path per clone leaks runtimes.
        if not hasattr(ranker, '_branch_chroma'):
            ranker._chroma_tmpdir = tempfile.TemporaryDirectory(prefix='memt-joint-chroma-')
            settings = VectorDBConfig()
            settings.path = ranker._chroma_tmpdir.name
            settings.embedding_model = ranker.model_path
            settings.hnsw_num_threads = 1
            ranker._branch_chroma = ChromaVectorDB(settings)
            ranker._chroma_view = {}
            ranker._chroma_vectors = {}
            ranker.chroma_stats = {'encoded_texts': 0, 'upserted_rows': 0, 'deleted_rows': 0,
                                   'embedding_seconds': 0., 'sync_seconds': 0.}
        self._chroma = ranker._branch_chroma
        # Rows are replaced, never edited in place by ListDB/JournalDB.
        self.collections = {name: dict(rows) for name, rows in collections.items()}

    def _vectors(self, texts):
        cache = self.ranker._chroma_vectors
        missing = list(dict.fromkeys(text for text in texts if text not in cache))
        if missing:
            started = time.perf_counter()
            cache.update(zip(missing, self._chroma.default_embedding_function(missing)))
            self.ranker.chroma_stats['encoded_texts'] += len(missing)
            self.ranker.chroma_stats['embedding_seconds'] += time.perf_counter() - started
        return [cache[text] for text in texts]

    def _sync(self, name):
        started = time.perf_counter()
        view = self.ranker._chroma_view
        if name not in view:
            self._chroma.create_collection(name, get_or_create=True)
            view[name] = {}
        previous, current = view[name], self.collections.get(name, {})
        removed = [key for key in previous if key not in current]
        changed = [key for key in current if previous.get(key) != current[key]]
        if removed:
            self._chroma.delete(name, ids=removed)
        if changed:
            documents = [current[key][0] for key in changed]
            self._chroma.upsert(name, ids=changed, documents=documents,
                                metadatas=[current[key][1] for key in changed],
                                embeddings=self._vectors(documents))
        view[name] = dict(current)
        self.ranker.chroma_stats['upserted_rows'] += len(changed)
        self.ranker.chroma_stats['deleted_rows'] += len(removed)
        self.ranker.chroma_stats['sync_seconds'] += time.perf_counter() - started

    def search(self, name, query_texts, top_k=5, **kwargs):
        # One synchronous runtime owns this view; switching branches first
        # removes invisible rows, including future memories in old snapshots.
        self._sync(name)
        return self._chroma.search(name, query_embeddings=self._vectors(query_texts), top_k=top_k,
                                   include=['documents', 'metadatas'])


class State:
    def __init__(self, sample_id, ranker):
        self.sample_id, self.ranker = sample_id, ranker
        self.journal = JournalDB(BranchDB(ranker))
        for kind in ('summary', 'personas', 'facts', 'experiences', 'turns'):
            self.journal.create_collection(self.collection(kind))
        self.sources, self.summary, self.personas = {}, '', {}
        self.summary_sources, self.persona_sources = set(), {}
        self.session, self.position, self.input_tokens, self.time = None, None, 0, ''

    def collection(self, kind):
        return self.sample_id + '_' + kind

    def enable_chroma(self):
        if isinstance(self.journal.db, ChromaBranchDB):
            return self
        self.journal.db = ChromaBranchDB(self.ranker, self.journal.db.collections)
        return self

    def clone(self):
        state = copy.copy(self)
        db = self.journal.db
        if isinstance(db, ChromaBranchDB):
            branch = ChromaBranchDB(self.ranker, db.collections)
        else:
            branch = BranchDB(self.ranker)
            branch.collections = {name: dict(rows) for name, rows in db.collections.items()}
        state.journal = JournalDB(branch)
        state.journal.versions = dict(self.journal.versions)
        state.journal.active = dict(self.journal.active)
        state.journal.operations = dict(self.journal.operations)
        state.sources, state.personas = dict(self.sources), dict(self.personas)
        state.summary_sources = set(self.summary_sources)
        state.persona_sources = {key: set(value) for key, value in self.persona_sources.items()}
        state.position = copy.copy(self.position)
        return state

    def to_dict(self):
        return copy.deepcopy(self._snapshot())

    def _snapshot(self):
        return {'sample_id': self.sample_id, 'collections': self.journal.db.collections,
            'backend': 'chroma' if isinstance(self.journal.db, ChromaBranchDB) else 'lexical',
            'versions': self.journal.versions, 'active': [[*key, value] for key, value in self.journal.active.items()],
            'operations': self.journal.operations, 'sources': self.sources, 'summary': self.summary,
            'personas': self.personas, 'summary_sources': sorted(self.summary_sources),
            'persona_sources': {k: sorted(v) for k, v in self.persona_sources.items()},
            'session': self.session, 'position': self.position, 'input_tokens': self.input_tokens, 'time': self.time}

    @classmethod
    def from_dict(cls, value, ranker):
        value = copy.deepcopy(value)
        state = cls(value['sample_id'], ranker)
        state.journal.db.collections = value['collections']
        state.journal.versions = value['versions']
        state.journal.active = {(a, b): c for a, b, c in value['active']}
        state.journal.operations = value.get('operations', {})
        for key in ('sources', 'summary', 'personas', 'session', 'position', 'input_tokens', 'time'):
            setattr(state, key, value[key])
        state.summary_sources = set(value['summary_sources'])
        state.persona_sources = {k: set(v) for k, v in value['persona_sources'].items()}
        if value.get('backend') == 'chroma':
            state.enable_chroma()
        return state

    def fingerprint(self):
        return digest(self._snapshot())

    def observe(self, session, block, position, count):
        if self.session != session['session_id']:
            self.summary, self.summary_sources = '', set()
        self.session, self.position = session['session_id'], list(position)
        self.time = session['time']
        for turn in block:
            if turn['turn_id'] in self.sources:
                raise ValueError('A source turn was observed twice')
            text = '%s at %s: %s' % (turn['speaker'], session['time'], turn['text'])
            self.sources[turn['turn_id']] = text
            self.input_tokens += count(text)

    def cost(self, count):
        rows = [row for collection in self.journal.db.collections.values() for row in collection.values()]
        # Documents include the original source copies written by the released tools.
        tokens = sum(count(row[0]) + sum(count(text) for text in row[1].get('original_turns', [])) for row in rows)
        ratio = tokens / max(1, self.input_tokens)
        return ratio / (1 + ratio)

    def statistics(self, count):
        tiers = {}
        for kind in ('summary', 'personas', 'facts', 'experiences', 'turns'):
            values = self.journal.db.collections[self.collection(kind)]
            tiers[kind] = {'items': len(values), 'document_tokens': sum(count(v[0]) for v in values.values()),
                           'source_copy_tokens': sum(count(t) for v in values.values() for t in v[1].get('original_turns', []))}
        raw_ids = {s for v in self.journal.db.collections[self.collection('turns')].values()
                   for s in flatten(v[1].get('source_turn_ids', []))}
        active = sum(v['document_tokens'] + v['source_copy_tokens'] for v in tiers.values())
        return {'tiers': tiers, 'active_tokens': active, 'input_tokens': self.input_tokens,
                'unbounded_memory_ratio': active / max(1, self.input_tokens),
                'raw_selected_turns': len(raw_ids), 'observed_turns': len(self.sources),
                'snapshot_bytes': len(json.dumps(self.to_dict(), ensure_ascii=False).encode('utf-8'))}


def builder_prompt(state, block):
    formation = MemoryFormation(None, 'paper-v2')
    messages = formation.construct_prompt(
        '\n'.join('[%s] %s' % (t['turn_id'], state.sources[t['turn_id']]) for t in block),
        state.summary, '\n'.join('Name: %s, Profile: %s' % p for p in state.personas.items()))
    return messages


def builder(state, block, completion, event):
    """Keep each valid Mem-T operation; audit invalid siblings separately."""
    branch = state.clone()
    formation = MemoryFormation(None, 'paper-v2')
    calls = parse_calls(completion.text)
    available = [t['turn_id'] for t in block]
    if not calls:
        raise InvalidAction('No complete formation calls')
    candidates, operations = [], []
    for index, call in enumerate(calls):
        tool = formation.tool_map.get(call['name'])
        valid = tool is not None and validate_call(call, tool)
        if call['name'] == 'create_raw':
            valid = valid and CreateRawTool.valid_ids(call['arguments'].get('turn_ids'), available)
        op = {'op_id': digest([event, index]), 'parent_operation_ids': [], 'source_turn_ids': available,
              'memory_refs': [], 'deleted_refs': [], 'call': call, 'role': 'B', 'position': branch.position, 'valid': valid}
        operations.append(op)
        if not valid:
            continue
        item = formation.execute_tool(call['name'], call['arguments'])
        if not item:
            op['valid'] = False
            continue
        if item['type'] in ('fact', 'experience'):
            candidates.append(dict(item, producer=op['op_id'], source_turn_ids=available, turn_time=branch.time))
            continue
        kind = {'summary': 'summary', 'persona': 'personas', 'raw': 'turns'}[item['type']]
        if kind == 'turns':
            ids = [t for t in available if t in item['turn_ids']]
            op['source_turn_ids'] = ids
            key, content = op['op_id'], '\n'.join(branch.sources[t] for t in ids)
        elif kind == 'summary':
            branch.summary_sources.update(available)
            ids, key, content = sorted(branch.summary_sources), branch.session, item['document']
            branch.summary = content
        else:
            key, content = item['name'], item['document']
            branch.personas[key] = content
            branch.persona_sources.setdefault(key, set()).update(available)
            ids = sorted(branch.persona_sources[key])
        metadata = {'source_turn_ids': ids, 'original_turns': [branch.sources[t] for t in ids], 'turn_time': branch.time}
        if kind != 'turns':
            metadata.update(memory_content=content, name=key,
                            updated_time='%08d:%08d' % tuple(branch.position))
        branch.journal.begin(op)
        if kind == 'turns':
            branch.journal.add(branch.collection(kind), [key], [content], [metadata])
        else:
            branch.journal.upsert(branch.collection(kind), [key], [content], [metadata])
        branch.journal.end()
    return branch, candidates, operations


def updater_prompt(state, item, top_k):
    kind = 'facts' if item['type'] == 'fact' else 'experiences'
    related = state.journal.search(state.collection(kind), [item['document']], top_k=top_k)
    candidates = [{'id': key, 'document': meta.get('memory_content', doc),
                   'turn_time': meta.get('turn_time', ''),
                   'start_time': meta.get('start_time', ''), 'end_time': meta.get('end_time', '')}
                  for key, doc, meta in zip(related['ids'][0], related['documents'][0], related['metadatas'][0])]
    tools = MemoryUpdate(None, state.journal)
    potential = {key: value for key, value in item.items() if key not in ('producer', 'source_turn_ids')}
    return tools.construct_prompt(potential, candidates, tools.get_tool_schemas(state.collection(kind))), candidates


def updater(state, item, related, completion, event):
    branch = state.clone()
    collection = branch.collection('facts' if item['type'] == 'fact' else 'experiences')
    evolution = MemoryUpdate(None, branch.journal)
    tools = {t.name: t for t in evolution._get_tools(collection)}
    calls = parse_calls(completion.text)
    if not calls:
        raise InvalidAction('No complete evolution calls')
    operations = []
    for index, call in enumerate(calls):
        tool = tools.get(call['name'])
        valid = tool is not None and validate_call(call, tool)
        if call['name'] in ('update_item', 'delete_item'):
            valid = valid and bool(branch.journal.get(collection, [call['arguments'].get('id', '')])['ids'])
        op = {'op_id': digest([event, index]), 'parent_operation_ids': [item['producer']],
              'source_turn_ids': item['source_turn_ids'], 'memory_refs': [], 'deleted_refs': [],
              'role': 'U', 'call': call, 'position': branch.position, 'valid': valid}
        operations.append(op)
        if not valid:
            continue
        text = '\n'.join('[%s] %s' % (t, branch.sources[t]) for t in item['source_turn_ids'])
        args = dict(call['arguments'], source_turn_ids=item['source_turn_ids'],
                    original_text=text, original_turns=[text],
                    op_ids={'formation': item['producer'], 'update': op['op_id']}, _memory_id=op['op_id'])
        branch.journal.begin(op)
        result = evolution.execute_tool(collection, call['name'], args)
        branch.journal.end()
        op['valid'] = result.get('status') == 'success'
    return branch, operations
