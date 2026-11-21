"""Compare incremental Chroma queries with Mem-T using real local BGE-M3."""
import json
import tempfile
from pathlib import Path

from experiments.io import write_json
from memory.branch import State, Ranker
from memory.settings import VectorDBConfig


def verify(model, output):
    root = Path(__file__).resolve().parents[1]
    reference = root.parent / 'Mem-T' / 'memory' / 'vector_db.py'
    namespace = {}
    source = reference.read_text().replace('from loguru import logger', 'from memory.logging import logger')
    exec(compile(source, str(reference), 'exec'), namespace)
    state = State('backend-audit', Ranker(model)).enable_chroma()
    name = state.collection('facts')
    documents = ['Jon is opening a dance studio.', 'Gina is reading The Lean Startup.',
                 'Jon lost his job as a banker on 19 January, 2023.',
                 'Gina went hiking in the mountains.', 'Jon visited Rome to clear his mind.',
                 'Gina got a tattoo a few years ago.']
    ids = ['fact-%d' % i for i in range(len(documents))]
    state.journal.add(name, ids, documents, [{'memory_content': doc, 'source_turn_ids': [key]}
                                           for key, doc in zip(ids, documents)])
    with tempfile.TemporaryDirectory(dir=output) as directory:
        settings = VectorDBConfig()
        settings.path, settings.embedding_model, settings.hnsw_num_threads = directory, model, 1
        db = namespace['ChromaVectorDB'](settings)
        db.create_collection(name)
        rows = state.journal.get(name)
        db.upsert(name, ids=rows['ids'], documents=rows['documents'], metadatas=rows['metadatas'])
        checks = []
        for query in ['What is Gina reading?', 'When did Jon lose his job?', 'Why did Jon visit Rome?']:
            for k in [1, 3, 5, 50]:
                left = state.journal.search(name, [query], top_k=k)
                right = db.search(name, query_texts=[query], top_k=k, include=['documents', 'metadatas'])
                assert all(left[key] == right[key] for key in ['ids', 'documents', 'metadatas']), (query, k)
                checks.append({'query': query, 'top_k': k, 'ids': left['ids'][0]})
        old = state.clone()
        state.journal.delete(name, ['fact-0'])
        state.journal.search(name, ['dance studio'])
        restored = old.journal.search(name, ['dance studio'])
        assert 'fact-0' in restored['ids'][0]
    result = {'passed': True, 'embedding_model': model, 'checks': checks,
              'scope': 'real BGE-M3; Chroma ranking/documents/metadata and branch switching'}
    write_json(Path(output) / 'BACKEND_PARITY.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(json.dumps(verify(args.model, args.output)))
