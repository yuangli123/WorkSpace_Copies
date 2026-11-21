"""Offline checks for phase usage, API failures, and measured resume."""

import json
import hashlib
import os
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from evaluation.measurement import Measurement, capture_openai, summarize


class MeasurementTests(unittest.TestCase):
    def test_sdk_usage_is_assigned_to_the_active_phase(self):
        class Chat:
            def create(self):
                return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=7, completion_tokens=3))

        class Embed:
            def create(self):
                return types.SimpleNamespace(usage=types.SimpleNamespace(prompt_tokens=5))

        fake = {}
        for name in ("openai", "openai.resources", "openai.resources.chat",
                     "openai.resources.chat.completions", "openai.resources.embeddings"):
            fake[name] = types.ModuleType(name)
        fake["openai.resources.chat.completions"].Completions = Chat
        fake["openai.resources.embeddings"].Embeddings = Embed
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, fake):
            answer = str(Path(temp) / "answers.json")
            meter = Measurement(answer)
            with capture_openai():
                with meter.phase(0, None, "conversation"):
                    with meter.phase(0, None, "construct"):
                        Chat().create()
                    with meter.phase(0, 0, "query"):
                        with meter.phase(0, 0, "retrieve"):
                            Embed().create()
                        with meter.phase(0, 0, "answer"):
                            Chat().create()
            meter.save_conversation(0)
            result = summarize(answer + ".metrics.json", {"0": 1})
            self.assertEqual(result["construct_llm_tokens_per_conversation"], 10)
            self.assertEqual(result["query_llm_tokens_per_question"], 10)
            self.assertEqual(result["tokens"]["query"]["embedding"]["prompt_tokens"], 5)
            self.assertEqual(result["calls"]["llm"]["calls"], 2)
            self.assertGreaterEqual(result["query_latency"]["mean_s"], 0)

    def test_missing_usage_failure_and_resume_do_not_invent_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            answer = str(Path(temp) / "answers.json")
            meter = Measurement(answer)
            with meter.phase(0, None, "construct"):
                meter.record(0, None, "construct", "llm", types.SimpleNamespace(usage=None), 0.1)
                meter.record(0, None, "construct", "llm", None, 0.1, failed=True)
            with meter.phase(0, 0, "query"):
                with meter.phase(0, 0, "answer"):
                    meter.record(0, 0, "answer", "llm", types.SimpleNamespace(
                        usage=types.SimpleNamespace(prompt_tokens=2, completion_tokens=1)), 0.1)
            meter.save_conversation(0)
            rows = {"0": [{"question": "q"}]}
            resumed = Measurement(answer, rows, resume=True)
            self.assertEqual(set(resumed.data["conversations"]), {"0"})
            result = summarize(answer + ".metrics.json", {"0": 1})
            self.assertIsNone(result["construct_llm_tokens_per_conversation"])
            self.assertEqual(result["calls"]["llm"]["failed"], 1)
            self.assertEqual(result["query_llm_tokens_per_question"], 3)
            Path(answer + ".metrics.json").unlink()
            with self.assertRaisesRegex(ValueError, "measurements are missing"):
                Measurement(answer, rows, resume=True)

    def test_extra_or_incomplete_questions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            answer = str(Path(temp) / "answers.json")
            meter = Measurement(answer)
            with meter.phase(0, None, "construct"):
                pass
            meter.save_conversation(0)
            with self.assertRaisesRegex(ValueError, "incomplete measurements"):
                Measurement(answer, {"0": [{"question": "q"}]}, resume=True)
            with self.assertRaisesRegex(ValueError, "Incomplete measurement questions"):
                summarize(answer + ".metrics.json", {"0": 1})

    def test_native_and_external_rows_remain_unchanged(self):
        from evaluation.protocol import load_selected
        from evaluation import external_runner
        from tests.test_comparison import load_runner
        from tests.test_external_protocol import locomo_sample
        _, runner = load_runner()

        class Native:
            def reset(self, *_):
                pass

            def add_memory(self, *_):
                pass

            def answer_question(self, *_args, **kwargs):
                with kwargs["measure_phase"]("retrieve"):
                    pass
                with kwargs["measure_phase"]("answer"):
                    pass
                return "ok"

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "locomo.json"
            data.write_text(json.dumps(locomo_sample()), encoding="utf-8")
            conversations, questions, _ = load_selected(str(data), "locomo")
            meter = Measurement(str(root / "native.json"))
            with meter.phase(0, None, "conversation"):
                native_rows, _ = runner.process_one_conversation(Native(), "recmem", 0,
                                                                 conversations[0], questions[0], 1, False,
                                                                 "locomo", meter=meter)
            meter.save_conversation(0)
            self.assertEqual(set(native_rows[0]),
                             {"question", "answer", "category", "response", "answerable"})

            class Adapter:
                memory = types.SimpleNamespace(memories={})

                def context(self, *_):
                    return ["memory"]

            env = {"OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
                   "EMBEDDING_API_BASE": "http://127.0.0.1:8001/v1", "PERSONAL_ROOT": ""}
            with patch.dict(os.environ, env), \
                 patch.object(external_runner, "source_version", return_value="v"), \
                 patch.object(external_runner, "capture_openai", return_value=nullcontext()), \
                 patch.object(external_runner, "build_adapter", return_value=Adapter()), \
                 patch.object(external_runner, "EmbeddingService", return_value=object()), \
                 patch.object(external_runner, "Answerer") as answerer:
                answerer.return_value.answer.return_value = ("ok", {"removed_chunks": 0})
                for method in ("memoryos", "mem0", "amem"):
                    answer = root / f"{method}.json"
                    external_runner.main(["--method", method, "--bench", "locomo", "--data_path", str(data),
                                          "--output_file", str(answer), "--store_root", str(root / method),
                                          "--measure"])
                    self.assertEqual(json.loads(answer.read_text(encoding="utf-8"))["0"], native_rows)
                    self.assertEqual(summarize(str(answer) + ".metrics.json", {"0": 1})["questions"], 1)

    def test_comparison_requires_complete_matching_measurements(self):
        from evaluation.compare_results import compare
        from evaluation.protocol import load_selected, question_text
        from tests.test_external_protocol import locomo_sample

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "locomo.json"
            data.write_text(json.dumps(locomo_sample()), encoding="utf-8")
            question = load_selected(str(data), "locomo")[1][0][0]
            answer = root / "answer.json"
            row = {"question": question_text(question), "answer": question.answer,
                   "category": question.category, "answerable": question.answerable,
                   "response": "drawer"}
            answer.write_text(json.dumps({"0": [row]}), encoding="utf-8")
            score = root / "score.json"
            score.write_text(json.dumps({"0": [{**row, "answer": str(row["answer"]),
                                                "llm_score": 1, "f1_score": 1,
                                                "bleu_score": 1}]}), encoding="utf-8")
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            (root / "answer.json.run.json").write_text(json.dumps({
                "method": "amem", "bench": "locomo", "model": "model",
                "llm_base_url": "http://localhost:8000/v1", "embedding_model": "embed",
                "embedding_base_url": "http://localhost:8001/v1", "dataset_sha256": digest,
                "conv_limit": None, "sample_size": None, "measure": True,
                "measurement_schema": 1}), encoding="utf-8")
            (root / "score.json.run.json").write_text(json.dumps({
                "bench": "locomo", "judge_model": "judge", "dataset_sha256": digest,
                "answer_file": os.path.realpath(answer),
                "answer_sha256": hashlib.sha256(answer.read_bytes()).hexdigest()}), encoding="utf-8")
            meter = Measurement(str(answer))
            with meter.phase(0, None, "construct"):
                pass
            with meter.phase(0, 0, "query"):
                with meter.phase(0, 0, "answer"):
                    pass
            meter.save_conversation(0)
            meter.save_judge(1.0)
            output = root / "comparison.json"
            compare("locomo", str(data), [("amem", str(answer), str(score))], str(output),
                    with_metrics=True)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["efficiency"][0]["questions"], 1)
            self.assertEqual(report["efficiency"][0]["judge_wall_s"], 1.0)
            (root / "answer.json.metrics.json").unlink()
            with self.assertRaises(FileNotFoundError):
                compare("locomo", str(data), [("amem", str(answer), str(score))],
                        str(root / "missing.json"), with_metrics=True)


if __name__ == "__main__":
    unittest.main()
