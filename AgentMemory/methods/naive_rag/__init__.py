"""Naive RAG baseline over contiguous 500-token history fragments."""

import uuid
from contextlib import nullcontext
from typing import Dict, List, Tuple

from recmem.embedding import OpenAIEmbedding
from recmem.vector_store import QdrantStore


class NaiveRAG:
    def __init__(
        self,
        answerer,
        embedder: OpenAIEmbedding,
        store_path: str,
        top_k: int,
        tokenizer,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.answerer = answerer
        self.embedder = embedder
        self.store = QdrantStore(embedding_dim=embedder.dim, path=store_path, exact_search=True)
        self.top_k = top_k
        self.tokenizer = tokenizer

    def index_conversation(self, conv_id: str, chunks: List[str]) -> int:
        collection = f"rag_{conv_id}"
        self.store.reset(collection)
        history = "\n".join(chunks)
        offsets = self.tokenizer.encode(history, add_special_tokens=False).offsets
        boundaries = [0] + [offsets[i][0] for i in range(500, len(offsets), 500)] + [len(history)]
        fragments = [history[start:end] for start, end in zip(boundaries, boundaries[1:]) if start < end]
        for start in range(0, len(fragments), 32):
            batch = fragments[start : start + 32]
            self.store.add_batch(
                embeddings=self.embedder.embed_batch(batch),
                payloads=batch,
                ids=[str(uuid.uuid4()) for _ in batch],
                collection_name=collection,
            )
        return len(fragments)

    def answer_question(
        self, question: str, conv_id: str, bench: str, measure_phase=None
    ) -> Tuple[str, Dict[str, object]]:
        with measure_phase("retrieve") if measure_phase else nullcontext():
            hits = self.store.search(
                q_embedding=self.embedder.embed(question),
                top_k=self.top_k,
                collection_name=f"rag_{conv_id}",
            )
        with measure_phase("answer") if measure_phase else nullcontext():
            return self.answerer.answer(
                [hit.content for hit in hits], question, bench, newest_first=True
            )
