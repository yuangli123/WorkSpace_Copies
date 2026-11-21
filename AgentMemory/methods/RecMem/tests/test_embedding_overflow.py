"""Embedding context overflow recovery without network tokenizer downloads."""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class BadRequest(Exception):
    status_code = 400


class FakeOpenAI:
    def __init__(self, **_):
        self.embeddings = self
        self.inputs = []
        self.fail_once = False

    def create(self, **request):
        texts = request["input"]
        self.inputs.append(list(texts))
        if self.fail_once:
            self.fail_once = False
            raise ConnectionError("temporary connection failure")
        if any(len(text) > 600 for text in texts):
            raise BadRequest(
                "This model's maximum context length is 2048 tokens. "
                "However, your request has 4105 input tokens."
            )
        return types.SimpleNamespace(data=[
            types.SimpleNamespace(index=i, embedding=[len(text)])
            for i, text in enumerate(texts)
        ])


class EmbeddingOverflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAI
        source = Path(__file__).resolve().parents[1] / "recmem" / "embedding" / "openai.py"
        spec = importlib.util.spec_from_file_location("embedding_under_test", source)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "EMPTY", "EMBEDDING_SEND_DIMENSIONS": "false"}):
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        cls.module = module
        cls.Embedding = module.OpenAIEmbedding

    def test_single_overflow_shrinks_without_tiktoken(self):
        embedding = self.Embedding(model="qwen3-embedding-0.6b", embedding_dims=1024)
        self.assertLess(embedding.embed("a" * 1000)[0], 600)
        self.assertEqual(len(embedding.client.inputs), 2)

    def test_batch_overflow_preserves_short_item(self):
        embedding = self.Embedding(model="qwen3-embedding-0.6b", embedding_dims=1024)
        vectors = embedding.embed_batch(["short", "b" * 1000])
        self.assertEqual(vectors[0], [5])
        self.assertLess(vectors[1][0], 600)

    def test_transient_error_retries_without_truncating(self):
        embedding = self.Embedding(model="qwen3-embedding-0.6b", embedding_dims=1024)
        embedding.client.fail_once = True
        with patch.object(self.module.time, "sleep"):
            self.assertEqual(embedding.embed("keep all text"), [13])
        self.assertEqual(embedding.client.inputs, [["keep all text"], ["keep all text"]])


if __name__ == "__main__":
    unittest.main()
