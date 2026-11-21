import copy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from evaluation.guided_parallel import _HNSW_REPAIR_PREVIOUS, validate_hnsw_recovery
from experiments.io import write_json
from memory.journal import JournalDB
from memory.vector_db import ChromaVectorDB


def test_hnsw_resume_allows_only_the_repair_source_delta():
    previous = {"config": {"seed": 42}, "inputs": {"q": "digest"},
                "code_sha256": {**_HNSW_REPAIR_PREVIOUS, "other.py": "same"}}
    current = copy.deepcopy(previous)
    current["code_sha256"].update({p: "repaired" for p in _HNSW_REPAIR_PREVIOUS})
    validate_hnsw_recovery(previous, current)
    for key in ("config", "inputs", "code_sha256"):
        unrelated = copy.deepcopy(current)
        unrelated[key]["other.py"] = "unrelated change"
        with pytest.raises(ValueError, match="original LoCoMo contract"):
            validate_hnsw_recovery(previous, unrelated)


def test_small_restored_hnsw_segments_reopen_in_a_new_process(tmp_path):
    chroma = pytest.importorskip("chromadb")

    class FakeEmbedding:
        def __call__(self, input):
            return [[1.0, float(index), 3.0] for index, _ in enumerate(input)]

    target = tmp_path / "chroma"
    db = ChromaVectorDB.__new__(ChromaVectorDB)
    db.config = SimpleNamespace(hnsw_num_threads=2)
    db.default_embedding_function = FakeEmbedding()
    db.client = chroma.PersistentClient(path=str(target))
    collections = {}
    for count in (1, 2, 65):
        ids = [str(i) for i in range(count)]
        collections[f"probe-{count}"] = {"ids": ids, "documents": ids,
                                         "metadatas": [{"memory_ref": value} for value in ids]}
    snapshot = tmp_path / "snapshot.json"
    write_json(snapshot, {"active": [], "versions": {}, "collections": collections})
    JournalDB.restore(snapshot, db)
    assert len(list(target.glob("*/index_metadata.pickle"))) == 3
    script = '''
from concurrent.futures import ThreadPoolExecutor
import sys
from chromadb import PersistentClient
client = PersistentClient(path=sys.argv[1])
collections = [client.get_collection(f"probe-{n}") for n in (1,2,65)]
def query(n):
    rows = collections[n % len(collections)].query(query_embeddings=[[1.0,0.0,3.0]], n_results=1)
    return rows["ids"]
with ThreadPoolExecutor(max_workers=8) as pool:
    assert list(pool.map(query, range(24))) == [[["0"]]] * 24
'''
    subprocess.run([sys.executable, "-B", "-c", script, str(target)], check=True)
