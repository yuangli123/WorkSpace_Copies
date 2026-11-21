"""Offline contract tests for the third-party adapters and shared comparison."""

import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.compare_results import compare, validate_rows
from evaluation.external_runner import AMemAdapter, Answerer, Mem0Adapter, MemoryOSAdapter, check_personal_workspace, main, source_version
from evaluation.protocol import load_selected, question_text


def locomo_sample():
    return [{
        "sample_id": "conv-one",
        "conversation": {
            "speaker_a": "Alice", "speaker_b": "Bob",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Alice", "text": "The key is in the drawer", "dia_id": "D1:1"},
                {"speaker": "Bob", "text": "I understand", "dia_id": "D1:2"},
            ],
        },
        "qa": [
            {"question": "Where is the key?", "answer": "drawer", "evidence": "D1:1", "category": 1},
            {"question": "Unknown?", "answer": "", "evidence": "", "category": 5},
        ],
    }]


def lme_sample(index, question_type="single-session-user", abstention=False):
    return {
        "question_id": f"question_{index}" + ("_abstention" if abstention else ""),
        "question_type": question_type,
        "question": "Where is the key?",
        "answer": "drawer",
        "answer_session_ids": [f"session_{index}"],
        "question_date": "2025-01-02",
        "haystack_dates": ["2025/01/01 (Wed) 10:00"],
        "haystack_session_ids": [f"session_{index}"],
        "haystack_sessions": [[{"role": "user", "content": "The key is in the drawer"},
                               {"role": "assistant", "content": "Noted"}]],
    }


class ExternalProtocolTests(unittest.TestCase):
    def test_scored_answer_uses_scorer_string_conversion(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "locomo.json"
            raw = locomo_sample()
            raw[0]["qa"][0]["answer"] = 2022
            data.write_text(json.dumps(raw), encoding="utf-8")
            question = load_selected(str(data), "locomo")[1][0][0]
            scores = Path(temp) / "scores.json"
            rows = {"0": [{"question": question_text(question), "answer": "2022",
                            "category": question.category, "response": "2022"}]}
            scores.write_text(json.dumps(rows), encoding="utf-8")
            self.assertEqual(validate_rows(str(scores), [[question]], scored=True), rows)
            with self.assertRaisesRegex(ValueError, "answer mismatch"):
                validate_rows(str(scores), [[question]])

    def test_source_version_uses_files_without_git(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for method, (folder, package) in {
                "memoryos": ("MemoryOS", "eval"),
                "mem0": ("mem0", "mem0"),
                "amem": ("A-mem", "agentic_memory"),
            }.items():
                source = root / folder / package
                source.mkdir(parents=True)
                file = source / "core.py"
                file.write_text("version = 1\n", encoding="utf-8")
                with patch("subprocess.run", side_effect=AssertionError("Git was called")):
                    first = source_version(method, root)
                    self.assertEqual(first, source_version(method, root))
                    file.write_text("version = 2\n", encoding="utf-8")
                    self.assertNotEqual(first, source_version(method, root))

    def test_six_methods_share_rows_on_both_benchmarks(self):
        from tests.test_comparison import Embedder, LLM, UnitTokenizer, load_runner

        baselines, runner = load_runner()

        class RecMem:
            def reset(self, _):
                pass

            def add_memory(self, *_):
                pass

            def answer_question(self, *_args, **_kwargs):
                return "ok"

        with tempfile.TemporaryDirectory() as temp:
            for bench, raw in (("locomo", locomo_sample()), ("longmemeval_s", [lme_sample(0)])):
                data = Path(temp) / f"{bench}.json"
                data.write_text(json.dumps(raw), encoding="utf-8")
                conversation, questions, _ = load_selected(str(data), bench)
                answerer = baselines.ContextAnswerer(LLM(), "model", 28000)
                native = {
                    "recmem": RecMem(),
                    "fullcontext": baselines.FullContext(answerer),
                    "naive_rag": baselines.NaiveRAG(answerer, Embedder(), str(Path(temp) / "rag"), 10,
                                                     UnitTokenizer()),
                }
                rows = [runner.process_one_conversation(memory, name, 0, conversation[0], questions[0],
                                                        1, False, bench)[0] for name, memory in native.items()]
                self.assertEqual(rows[0], rows[1])
                self.assertEqual(rows[1], rows[2])
                fake_answerer = types.SimpleNamespace(answer=lambda *_: ("ok", {"removed_chunks": 0}))
                env = {"OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
                       "EMBEDDING_API_BASE": "http://127.0.0.1:8001/v1", "PERSONAL_ROOT": ""}
                with patch.dict(os.environ, env), patch("evaluation.external_runner.source_version", return_value="v"), \
                     patch("evaluation.external_runner.EmbeddingService", return_value=object()), \
                     patch("evaluation.external_runner.Answerer", return_value=fake_answerer), \
                     patch("evaluation.external_runner.build_adapter", return_value=types.SimpleNamespace(
                         context=lambda *_: ["memory"])):
                    for method in ("memoryos", "mem0", "amem"):
                        output = Path(temp) / f"{bench}_{method}.json"
                        main(["--method", method, "--bench", bench, "--data_path", str(data),
                              "--output_file", str(output), "--store_root", str(Path(temp) / f"store_{method}")])
                        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["0"], rows[0])

    def test_shared_scorer_records_judge_provenance(self):
        from tests.test_comparison import load_runner

        _, runner = load_runner()
        with tempfile.TemporaryDirectory() as temp:
            answer = Path(temp) / "answers.json"
            score = Path(temp) / "scores.json"
            answer.write_text(json.dumps({"0": [{"question": "Where?", "answer": "drawer", "category": 1,
                                                 "response": "drawer", "answerable": True}]}), encoding="utf-8")
            (Path(str(answer) + ".run.json")).write_text(json.dumps({"dataset_sha256": "hash"}), encoding="utf-8")
            with patch.object(runner, "generate_eval_summary"):
                runner.eval_experiment(str(answer), str(score), "locomo", "judge")
            manifest = json.loads(Path(str(score) + ".run.json").read_text(encoding="utf-8"))
            self.assertEqual((manifest["judge_model"], manifest["dataset_sha256"]), ("judge", "hash"))
            self.assertEqual(manifest["answer_sha256"], hashlib.sha256(answer.read_bytes()).hexdigest())

    def test_personal_workspace_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp) / "workspace"
            workspace.mkdir()
            with patch.dict(os.environ, {"PERSONAL_ROOT": temp}):
                check_personal_workspace([workspace / "RecMem"])
                with self.assertRaisesRegex(ValueError, "Path must stay"):
                    check_personal_workspace([Path(temp) / "other"])

    def test_method_adapters_ingest_without_answers(self):
        conversation_data = locomo_sample()
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "locomo.json"
            data.write_text(json.dumps(conversation_data), encoding="utf-8")
            conversation = load_selected(str(data), "locomo")[0][0]
            added = []

            class Short:
                def __init__(self, **_):
                    self.items = []

                def add_qa_pair(self, item):
                    self.items.append(item)

                def is_full(self):
                    return True

                def get_all(self):
                    return self.items

            class Long:
                def __init__(self, **_):
                    pass

                def get_user_profile(self, _):
                    return None

                def get_assistant_knowledge(self):
                    return []

            class Retriever:
                def __init__(self, *_args, **_kwargs):
                    pass

                def retrieve(self, *_args, **_kwargs):
                    return {"retrieval_queue": [], "long_term_knowledge": []}

            def module(name, **attrs):
                item = types.ModuleType(name)
                item.__dict__.update(attrs)
                return item

            fake_modules = {
                "short_term_memory": module("short_term_memory", ShortTermMemory=Short),
                "mid_term_memory": module("mid_term_memory", MidTermMemory=lambda **_: object()),
                "long_term_memory": module("long_term_memory", LongTermMemory=Long),
                "dynamic_update": module("dynamic_update", DynamicUpdate=lambda *_, **__: types.SimpleNamespace(
                    bulk_evict_and_update_mid_term=lambda: None)),
                "retrieval_and_answer": module("retrieval_and_answer", RetrievalAndAnswer=Retriever),
                "main_loco_parse": module("main_loco_parse", update_user_profile_from_top_segment=lambda *args: None),
                "utils": module("utils", OpenAIClient=lambda: object()),
            }
            with patch.dict(sys.modules, fake_modules):
                memoryos = MemoryOSAdapter(Path(temp), Path(temp) / "memoryos", conversation, "locomo", "conv-one", 10)
            self.assertIn("key is in the drawer", memoryos.context("Where?", 10)[0])

            class Memory:
                @classmethod
                def from_config(cls, config):
                    added.append(config)
                    return cls()

                def add(self, messages, **kwargs):
                    added.append((messages, kwargs))

                def search(self, *_args, **_kwargs):
                    return {"results": [{"memory": "drawer"}]}

            with patch.dict(sys.modules, {"mem0": module("mem0", Memory=Memory)}):
                mem0 = Mem0Adapter(Path(temp), Path(temp) / "mem0store", conversation, "locomo", "conv-one",
                                   "model", "embed", 1024, "http://llm", "http://embed")
            self.assertEqual(os.environ["MEM0_DIR"], str(Path(temp) / "runtime"))
            self.assertEqual(mem0.context("Where?", 10), ["drawer"])
            self.assertNotIn("timestamp", added[1][1])
            self.assertIn("session_timestamp", added[1][1]["metadata"])
            self.assertTrue(str(added[0]["history_db_path"]).startswith(temp))
            self.assertNotIn("embedding_dims", added[0]["embedder"]["config"])
            self.assertEqual(added[0]["vector_store"]["config"]["embedding_model_dims"], 1024)

            class Agentic:
                def __init__(self, **kwargs):
                    added.append(kwargs)

                def add_note(self, content, time):
                    added.append((content, time))

                def search_agentic(self, *_args, **_kwargs):
                    return [{"content": "drawer"}]

            with patch.dict(sys.modules, {
                "agentic_memory": module("agentic_memory"),
                "agentic_memory.memory_system": module("agentic_memory.memory_system", AgenticMemorySystem=Agentic),
            }):
                amem = AMemAdapter(Path(temp), conversation, "locomo", "model", object())
            self.assertEqual(amem.context("Where?", 10), ["drawer"])
            self.assertEqual(added[-1][1], "202305081356")
            lme = Path(temp) / "lme.json"
            lme.write_text(json.dumps([lme_sample(0)]), encoding="utf-8")
            lme_conversation = load_selected(str(lme), "longmemeval_s")[0][0]
            with patch.dict(sys.modules, fake_modules):
                lme_memoryos = MemoryOSAdapter(Path(temp), Path(temp) / "memoryos_lme", lme_conversation,
                                               "longmemeval_s", "question_0", 10)
            self.assertIn("[user]", lme_memoryos.context("Where?", 10)[0])
            with patch.dict(sys.modules, {
                "agentic_memory": module("agentic_memory"),
                "agentic_memory.memory_system": module("agentic_memory.memory_system", AgenticMemorySystem=Agentic),
            }):
                AMemAdapter(Path(temp), lme_conversation, "longmemeval_s", "model", object())
            self.assertEqual(added[-1][1], "202501011000")

    def test_selected_questions_and_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            locomo = Path(temp) / "locomo.json"
            locomo.write_text(json.dumps(locomo_sample()), encoding="utf-8")
            convs, questions, ids = load_selected(str(locomo), "locomo")
            self.assertEqual((len(convs), len(questions[0]), ids), (1, 1, [["conv-one:0"]]))
            lme = Path(temp) / "lme.json"
            lme.write_text(json.dumps([lme_sample(0), lme_sample(1, abstention=True)]), encoding="utf-8")
            _, questions, ids = load_selected(str(lme), "longmemeval_s")
            self.assertEqual(ids, [["question_0"], ["question_1_abstention"]])
            self.assertFalse(questions[1][0].answerable)
            self.assertIn("The date of the question is 2025-01-02", question_text(questions[0][0]))

    def test_lme_sample_matches_original_runner(self):
        from tests.test_comparison import load_runner

        _, runner = load_runner()
        names = ("single-session-user", "single-session-assistant", "single-session-preference",
                 "temporal-reasoning", "knowledge-update", "multi-session")
        raw = []
        for index, name in enumerate(names):
            for offset in range(3):
                item = lme_sample(index * 3 + offset, name)
                item["question"] = f"Question {index * 3 + offset}?"
                raw.append(item)
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "lme.json"
            data.write_text(json.dumps(raw), encoding="utf-8")
            all_convs, all_questions = runner.ConvLoader.load_longmemeval_s(str(data))
            _, original = runner.sample_lme_by_question_type(all_convs, all_questions, 1)
            _, selected, ids = load_selected(str(data), "longmemeval_s", sample_size=1)
            self.assertEqual([group[0].content for group in selected],
                             [group[0].content for group in original])
            self.assertEqual(len(ids), 6)

    def test_context_budget_removes_low_priority_memory(self):
        answerer = object.__new__(Answerer)
        answerer.model = "model"
        answerer.budget = 170
        captured = []

        def create(**kwargs):
            captured.append(kwargs["messages"][0]["content"])
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="drawer"))])

        answerer.client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)))
        response, stats = answerer.answer(["high priority", "low priority " * 100], "Where?", "locomo")
        self.assertEqual(response, "drawer")
        self.assertGreater(stats["removed_tokens_estimate"], 0)
        self.assertIn("high priority", captured[-1])
        self.assertNotIn("low priority", captured[-1])

    def test_interruption_resume_and_comparison_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "lme.json"
            data.write_text(json.dumps([lme_sample(0), lme_sample(1)]), encoding="utf-8")
            answers = root / "answers.json"
            store = root / "store"
            args = ["--method", "amem", "--bench", "longmemeval_s", "--data_path", str(data),
                    "--output_file", str(answers), "--store_root", str(store)]
            calls = []

            def adapter(_args, _source, _store, _conversation, _bench, index, *_):
                calls.append(index)
                if index == 1 and len(calls) == 2:
                    raise RuntimeError("interrupted")
                return types.SimpleNamespace(context=lambda *_: ["drawer"])

            fake_answerer = types.SimpleNamespace(answer=lambda *_: ("drawer", {"removed_chunks": 0}))
            env = {"OPENAI_API_BASE": "http://127.0.0.1:8000/v1",
                   "EMBEDDING_API_BASE": "http://127.0.0.1:8001/v1"}
            with patch.dict(os.environ, env), patch("evaluation.external_runner.source_version", return_value="version"), \
                 patch("evaluation.external_runner.EmbeddingService", return_value=object()), \
                 patch("evaluation.external_runner.Answerer", return_value=fake_answerer), \
                 patch("evaluation.external_runner.build_adapter", side_effect=adapter):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    main(args)
                self.assertEqual(list(json.loads(answers.read_text(encoding="utf-8"))), ["0"])
                main(args + ["--resume"])
                self.assertEqual(calls, [0, 1, 1])
                with self.assertRaisesRegex(ValueError, "parameters changed"):
                    main(args + ["--resume", "--top_k", "9"])

            rows = json.loads(answers.read_text(encoding="utf-8"))
            self.assertEqual(set(rows["0"][0]), {"question", "answer", "category", "response", "answerable"})
            scores = root / "scores.json"
            scores.write_text(json.dumps({key: [{**item, "llm_score": 1, "f1_score": 1, "bleu_score": 1}
                                               for item in group] for key, group in rows.items()}), encoding="utf-8")
            (root / "scores.json.run.json").write_text(json.dumps({
                "bench": "longmemeval_s", "judge_model": "judge",
                "dataset_sha256": json.loads((root / "answers.json.run.json").read_text())["dataset_sha256"],
                "answer_file": os.path.realpath(answers),
                "answer_sha256": hashlib.sha256(answers.read_bytes()).hexdigest(),
            }), encoding="utf-8")
            output = root / "comparison.json"
            table = compare("longmemeval_s", str(data), [("amem", str(answers), str(scores))], str(output))
            self.assertEqual(table[0]["count"], 2)
            scored_rows = json.loads(scores.read_text(encoding="utf-8"))
            scored_rows["0"][0]["response"] = "stale response"
            scores.write_text(json.dumps(scored_rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Scored response differs"):
                compare("longmemeval_s", str(data), [("amem", str(answers), str(scores))],
                        str(root / "stale_scores.json"))
            rows["1"][0]["answer"] = "wrong"
            answers.write_text(json.dumps(rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Answer changed after scoring"):
                compare("longmemeval_s", str(data), [("amem", str(answers), str(scores))],
                        str(root / "bad.json"))


if __name__ == "__main__":
    unittest.main()
