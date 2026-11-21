import gc

import pytest

from memory.branch import ChromaBranchDB, Ranker, State
from memory.journal import ListDB


class Embedding:
    def __init__(self):
        self.texts = []

    def __call__(self, input):
        self.texts.extend(input)
        return [[1., 0.] if 'original' in text else [0., 1.] for text in input]


def test_clones_do_not_write_or_encode_until_search(monkeypatch):
    import memory.vector_db
    instances = []

    class Store(ListDB):
        def __init__(self, settings):
            super().__init__()
            self.default_embedding_function = Embedding()
            assert settings.hnsw_num_threads == 1
            instances.append(self)

        def search(self, name, query_embeddings, **kwargs):
            return {key: [value] for key, value in self.get(name).items()}

    monkeypatch.setattr(memory.vector_db, 'ChromaVectorDB', Store)
    state = State('episode', Ranker(fixture=True)).enable_chroma()
    name = state.collection('facts')
    state.journal.upsert(name, ['fact'], ['original'], [{'source_turn_ids': ['t1']}])
    store = instances[0]
    snapshot = state.to_dict()
    for _ in range(100):
        clone = state.clone()
        clone.journal.upsert(name, ['fact'], ['changed'], [{'source_turn_ids': ['t2']}])
        assert isinstance(clone.journal.db, ChromaBranchDB)
        assert state.to_dict() == snapshot
        assert clone.journal.get(name)['documents'] == ['changed']
    assert len(instances) == 1 and store.collections == {}
    assert store.default_embedding_function.texts == []
    clone.journal.search(name, ['changed'])
    state.journal.search(name, ['original'])
    clone.journal.search(name, ['changed'])
    assert store.default_embedding_function.texts == ['changed', 'original']
    restored = State.from_dict(snapshot, state.ranker)
    assert restored.to_dict() == snapshot


def test_real_chroma_branch_switching_resources_and_search(monkeypatch):
    pytest.importorskip('chromadb')
    from chromadb.api.shared_system_client import SharedSystemClient
    from chromadb.utils import embedding_functions
    from pathlib import Path

    encoder = Embedding()
    monkeypatch.setattr(embedding_functions, 'SentenceTransformerEmbeddingFunction', lambda **kw: encoder)
    state = State('episode', Ranker(fixture=True)).enable_chroma()
    name = state.collection('facts')
    state.journal.upsert(name, ['fact'], ['original'], [{'source_turn_ids': ['t1']}])
    old = state.to_dict()
    state.journal.search(name, ['original'], top_k=1)
    initial_systems = len(SharedSystemClient._identifier_to_system)
    threads = lambda: len(list(Path('/proc/self/task').iterdir())) if Path('/proc/self/task').is_dir() else 0
    initial_threads = threads()
    for _ in range(40):
        clone = state.clone()
        clone.journal.upsert(name, ['future'], ['changed'], [{'source_turn_ids': ['t2']}])
        clone.journal.delete(name, ['fact'])
        assert clone.journal.search(name, ['changed'], top_k=5)['ids'] == [['future']]
        result = state.journal.search(name, ['original'], top_k=5)
        assert result['ids'] == [['fact']]
        assert result['metadatas'][0][0]['source_turn_ids'] == ['t1']
        del clone
    later = state.clone()
    later.journal.upsert(name, ['future'], ['changed'], [{'source_turn_ids': ['t2']}])
    later.journal.search(name, ['changed'], top_k=5)
    historical = State.from_dict(old, state.ranker)
    assert historical.journal.search(name, ['original'], top_k=5)['ids'] == [['fact']]
    gc.collect()
    client = state.journal.db._chroma.client
    assert len(SharedSystemClient._identifier_to_system) == initial_systems
    assert client.count_collections() == 1
    assert encoder.texts == ['original', 'changed']
    assert threads() <= initial_threads + 8
    print({'clients': 1, 'clones': 40, 'encoded_texts': len(encoder.texts),
           'threads_before': initial_threads, 'threads_after': threads()})
