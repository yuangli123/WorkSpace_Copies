"""Real BGE-M3 comparison against the saved pre-fix branch implementation."""
import argparse
import importlib.util
import json
import time
from pathlib import Path

from memory import branch
from memory.settings import VectorDBConfig
from memory.vector_db import ChromaVectorDB


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('reference_branch', args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model = '/public/home/wwxia/yuang.li/models/bge-m3'
    documents = ['Person %d enjoys hiking and works as a science teacher in Seattle.' % i for i in range(16)]
    results = {}
    for label, module in [('before', reference), ('after', branch)]:
        started = time.perf_counter()
        ranker = module.Ranker(model)
        state = module.State('benchmark', ranker).enable_chroma()
        name = state.collection('facts')
        state.journal.upsert(name, [str(i) for i in range(len(documents))], documents,
                             [{'source_turn_ids': ['t%d' % i]} for i in range(len(documents))])
        encoder = state.journal.db._chroma.default_embedding_function
        counter = {'texts': 0}

        class Counted:
            def __init__(self, encoder, counter):
                self.encoder, self.counter = encoder, counter

            def __call__(self, input):
                self.counter['texts'] += len(input)
                return self.encoder(input)

            def __getattr__(self, name):
                return getattr(self.encoder, name)

        state.journal.db._chroma.default_embedding_function = Counted(encoder, counter)
        query = 'Who works as a science teacher?'
        # Warm encoding/index state before measuring the repeated-clone path.
        state.journal.search(name, [query], top_k=5)
        warmup_seconds = time.perf_counter() - started
        counter['texts'] = 0
        started = time.perf_counter()
        for i in range(3):
            cloned = state.clone()
            cloned.journal.upsert(name, ['new'], ['Person %d moved to Boston.' % i],
                                  [{'source_turn_ids': ['new%d' % i]}])
            assert state.journal.get(name)['ids'] == [str(k) for k in range(16)]
            assert len(cloned.journal.search(name, [query], top_k=5)['ids'][0]) == 5
        results[label] = {'warmup_seconds': warmup_seconds,
                          'clone_update_search_seconds': time.perf_counter() - started,
                          'encoded_texts_in_measured_phase': counter['texts']}
        if label == 'after':
            # Compare IDs/order/metadata against a freshly built plain Mem-T DB.
            settings = VectorDBConfig(embedding_model=model, path=str(Path(args.output).parent / 'plain-chroma'))
            settings.hnsw_num_threads = 1
            plain = ChromaVectorDB(settings)
            rows = cloned.journal.get(name)
            plain.create_collection('alignment_check', get_or_create=True)
            plain.upsert('alignment_check', **rows)
            expected = plain.search('alignment_check', query_texts=[query], top_k=5)
            actual = cloned.journal.search(name, [query], top_k=5)
            assert actual['ids'] == expected['ids']
            assert actual['metadatas'] == expected['metadatas']
            results[label]['plain_chroma_topk_metadata_match'] = True
    assert results['after']['encoded_texts_in_measured_phase'] == 3
    results['speedup'] = results['before']['clone_update_search_seconds'] / results['after']['clone_update_search_seconds']
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
