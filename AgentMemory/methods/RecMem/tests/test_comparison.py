"""Offline checks for the comparison runner (no model or Qdrant service needed)."""

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class Store:
    def __init__(self, **_):
        self.collections = {}

    def reset(self, name):
        self.collections[name] = []

    def add_batch(self, *, embeddings, payloads, ids, collection_name):
        self.collections.setdefault(collection_name, []).extend(payloads)

    def search(self, *, q_embedding, top_k, collection_name):
        contents = self.collections.get(collection_name, [])
        ranked = sorted(contents, key=lambda text: q_embedding[0] in text, reverse=True)
        return [types.SimpleNamespace(content=text) for text in ranked[:top_k]]


class Embedder:
    dim = 1

    def embed(self, text):
        return [text]

    def embed_batch(self, texts):
        return [[text] for text in texts]


class UnitTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return types.SimpleNamespace(offsets=[(i, i + 1) for i in range(len(text))])


class LLM:
    def __init__(self, max_chars=None):
        self.max_chars = max_chars
        self.prompts = []
        self.requests = []

    def chat_completion(self, **kwargs):
        self.requests.append(kwargs)
        prompt = kwargs["messages"][0].content
        self.prompts.append(prompt)
        if self.max_chars and len(prompt) > self.max_chars:
            raise ValueError("maximum context length exceeded token limit")
        return types.SimpleNamespace(content="ok")


def module(name, **attributes):
    value = types.ModuleType(name)
    value.__dict__.update(attributes)
    return value


def load_runner():
    """Load production modules with small provider stubs for this offline test."""
    fake_modules = {
        "dotenv": module("dotenv", load_dotenv=lambda: None),
        "pandas": module("pandas"),
        "tqdm": module("tqdm", tqdm=lambda values, **_: values),
        "recmem.embedding": module("recmem.embedding", OpenAIEmbedding=Embedder),
        "recmem.llm": module("recmem.llm", OpenAIClient=LLM),
        "recmem.llm.base": module(
            "recmem.llm.base", Message=lambda role, content: types.SimpleNamespace(role=role, content=content)
        ),
        "recmem.prompt": module("recmem.prompt", ANSWER_PROMPT_LOCOMO="locomo", ANSWER_PROMPT_LME="lme"),
        "recmem.rec_mem": module("recmem.rec_mem", RecMem=object, RecMemConfig=object),
        "recmem.vector_store": module("recmem.vector_store", QdrantStore=Store, VectorStore=object),
        "evaluation.metrics.llm_judge": module(
            "evaluation.metrics.llm_judge",
            evaluate_llm_judge=lambda *_, **__: 1,
            evaluate_llm_judge_lme=lambda *_, **__: 1,
        ),
        "evaluation.metrics.utils": module(
            "evaluation.metrics.utils",
            calculate_bleu_scores=lambda *args: {"bleu1": 1},
            calculate_metrics=lambda *args: {"f1": 1},
        ),
    }
    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("evaluation.baselines", None)
        sys.modules.pop("evaluation.run_experiments", None)
        baselines = importlib.import_module("evaluation.baselines")
        runner = importlib.import_module("evaluation.run_experiments")
    return baselines, runner


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baselines, cls.runner = load_runner()

    def test_native_run_prepares_manifest_without_git(self):
        args = types.SimpleNamespace(
            output_file="answers.json", bench="locomo", eval_only=False,
            resume=False, overwrite=False, enable_stat=False,
            method="fullcontext", search_only=False, rag_store=None,
            context_budget=28000, top_k=10, conv_limit=1, sample_size=None,
            model="model", judge_model="judge",
        )
        with patch.object(self.runner, "add_arguments", return_value=args), \
             patch.object(self.runner, "prepare_run", side_effect=RuntimeError("manifest reached")), \
             patch("subprocess.run", side_effect=AssertionError("Git was called")), \
             patch.dict(os.environ, {"LOCOMO_PATH": "dataset.json", "LOCOMO_SCORE_PATH": "scores"}):
            with self.assertRaisesRegex(RuntimeError, "manifest reached"):
                self.runner.main()

    def test_fullcontext_never_truncates_history(self):
        complete_llm = LLM()
        response, stats = self.baselines.FullContext(
            self.baselines.ContextAnswerer(complete_llm, "model", 1)
        ).answer_question("Where?", "locomo", ["old " * 80, "new " * 50])
        self.assertEqual(response, "ok")
        self.assertEqual(stats["context_mode"], "fullcontext")
        self.assertEqual(stats["removed_chunks"], 0)
        self.assertIn("old old", complete_llm.prompts[-1])
        self.assertIn("new new", complete_llm.prompts[-1])
        self.assertEqual(complete_llm.requests[-1]["max_tokens"], 512)

        llm = LLM(max_chars=500)
        answerer = self.baselines.ContextAnswerer(llm, "model", 700)
        with self.assertRaisesRegex(ValueError, "maximum context length"):
            self.baselines.FullContext(answerer).answer_question(
                "Where?", "locomo", ["old " * 80, "new " * 50]
            )
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("old old", llm.prompts[-1])
        self.assertIn("new", llm.prompts[-1])

    def test_rag_top_k_and_separate_collection(self):
        llm = LLM()
        rag = self.baselines.NaiveRAG(
            self.baselines.ContextAnswerer(llm, "model", 2000), Embedder(), "own/rag", 3,
            UnitTokenizer(),
        )
        history = "a" * 500 + "b" * 500 + "target" + "c" * 494 + "d" * 500
        count = rag.index_conversation("locomo_conv_0", [history])
        self.assertEqual(count, 4)
        self.assertEqual("".join(rag.store.collections["rag_locomo_conv_0"]),
                         history)
        self.assertTrue(all(len(fragment) <= 500 for fragment in
                            rag.store.collections["rag_locomo_conv_0"]))
        answer, info = rag.answer_question("target", "locomo_conv_0", "locomo")
        self.assertEqual(answer, "ok")
        self.assertEqual(info["context_mode"], "naive_rag")
        self.assertIn("target", llm.prompts[-1])
        self.assertNotIn("d" * 100, llm.prompts[-1])
        self.assertEqual(llm.requests[-1]["max_tokens"], 512)

    def test_three_methods_share_locomo_rows(self):
        speaker = lambda role, content: types.SimpleNamespace(
            speaker=role, to_string=lambda: f"[{role}]: {content}"
        )
        conversation = types.SimpleNamespace(sessions=[types.SimpleNamespace(
            date_time="2025-01-01T00:00:00",
            messages=[speaker("A", "where is the key"), speaker("B", "in the drawer")],
        )])
        question = [types.SimpleNamespace(
            content="Where is the key?", answer="drawer", category=1,
            date=None, answerable=True,
        )]

        class RecMem:
            def reset(self, _):
                self.messages = []

            def add_memory(self, message, timestamp, _):
                self.messages.append((message, timestamp))

            def answer_question(self, *_args, **_kwargs):
                return "ok"

        answerer = self.baselines.ContextAnswerer(LLM(), "model", 2000)
        methods = {
            "recmem": RecMem(),
            "fullcontext": self.baselines.FullContext(answerer),
            "naive_rag": self.baselines.NaiveRAG(answerer, Embedder(), "own/rag", 1, UnitTokenizer()),
        }
        rows = []
        for name, method in methods.items():
            result, _ = self.runner.process_one_conversation(
                method, name, 0, conversation, question, 1, False, "locomo"
            )
            rows.append(result[0])
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[1], rows[2])
        self.assertEqual(methods["recmem"].messages[0][0],
                         "[A]: where is the key\n[B]: in the drawer")
        for row in rows:
            self.assertEqual(self.runner.process_item(("0", [row]), "locomo", "judge")["0"][0]["llm_score"], 1)
        with tempfile.TemporaryDirectory() as directory:
            answers = os.path.join(directory, "answers.json")
            scores = os.path.join(directory, "scores.json")
            with open(answers, "w") as handle:
                json.dump({"0": rows}, handle)
            with patch.object(self.runner, "generate_eval_summary"):
                self.runner.eval_experiment(answers, scores, "locomo", "judge")
            with open(scores) as handle:
                evaluated = json.load(handle)
            self.assertEqual(len(evaluated["0"]), 3)

    def test_shared_output_and_resume(self):
        loader = self.runner.ConvLoader
        message = lambda speaker, content: types.SimpleNamespace(
            speaker=speaker, to_string=lambda: f"[{speaker}]: {content}"
        )
        convs = [
            types.SimpleNamespace(sessions=[types.SimpleNamespace(
                date_time="2025-01-01T00:00:00", messages=[message("user", "hello"), message("assistant", "hi")]
            )]) for _ in range(2)
        ]
        questions = [[types.SimpleNamespace(
            content="hello?", answer="hi", category=0, date="2025-01-02", answerable=True
        )] for _ in range(2)]

        class RecMem:
            token_monitor = None

            def __init__(self, fail=False):
                self.fail = fail
                self.seen = []

            def reset(self, conv_id):
                self.seen.append(conv_id)
                if self.fail and conv_id.endswith("_1"):
                    raise RuntimeError("interrupted")

            def add_memory(self, *_):
                pass

            def answer_question(self, *_args, **_kwargs):
                return "hi"

        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "dataset.json")
            output = os.path.join(directory, "answers.json")
            with open(data_path, "w") as handle:
                handle.write("[]")
            config = {"method": "recmem", "model": "model"}
            with self.assertRaises(ValueError):
                self.runner.prepare_run(data_path, data_path, config, False, True)
            self.runner.prepare_run(output, data_path, config, False, False)
            with patch.object(loader, "load_longmemeval_s", return_value=(convs, questions)):
                with self.assertRaises(RuntimeError):
                    self.runner.process_all_conversations(
                        RecMem(fail=True), "recmem", data_path, output, "longmemeval_s", 1,
                        question_max_workers=1,
                    )
                with open(output) as handle:
                    saved = json.load(handle)
                self.assertEqual(list(saved), ["0"])
                self.runner.prepare_run(output, data_path, config, True, False)
                resumed = RecMem()
                self.runner.process_all_conversations(
                    resumed, "recmem", data_path, output, "longmemeval_s", 1,
                    question_max_workers=1, resume=True,
                )
                self.assertEqual(resumed.seen, ["longmemeval_s_conv_1"])
                with open(output) as handle:
                    rows = json.load(handle)
                self.assertEqual(set(rows), {"0", "1"})
                self.assertEqual(set(rows["0"][0]), {
                    "question", "answer", "category", "response", "answerable"
                })
                self.assertIn("The date of the question is 2025-01-02", rows["0"][0]["question"])
                self.assertEqual(self.runner.process_item(("0", rows["0"]), "longmemeval_s", "judge")["0"][0]["llm_score"], 1)
                with self.assertRaises(ValueError):
                    self.runner.prepare_run(output, data_path, {**config, "model": "other"}, True, False)
                with open(data_path, "w") as handle:
                    handle.write("[1]")
                with self.assertRaises(ValueError):
                    self.runner.prepare_run(output, data_path, config, True, False)

    def test_lme_baselines_keep_question_date_and_turns(self):
        role = lambda speaker, content: types.SimpleNamespace(
            speaker=speaker, to_string=lambda: f"[{speaker}]: {content}"
        )
        conversation = types.SimpleNamespace(sessions=[types.SimpleNamespace(
            date_time="2025-01-01 10:00 (Wed)",
            messages=[role("assistant", "earlier"), role("user", "key?"), role("assistant", "drawer")],
        )])
        question = [types.SimpleNamespace(
            content="Where?", answer="drawer", category=0,
            date="2025-01-03", answerable=True,
        )]
        chunks, dates = self.runner.conversation_chunks(conversation, "longmemeval_s")
        self.assertEqual(chunks, ["[assistant]: earlier", "[user]: key?\n[assistant]: drawer"])
        self.assertEqual(dates, ["2025-01-01 10:00 (Wed)"] * 2)
        answerer = self.baselines.ContextAnswerer(LLM(), "model", 2000)
        for method, memory in (
            ("fullcontext", self.baselines.FullContext(answerer)),
            ("naive_rag", self.baselines.NaiveRAG(answerer, Embedder(), "own/lme", 2, UnitTokenizer())),
        ):
            rows, _ = self.runner.process_one_conversation(
                memory, method, 0, conversation, question, 1, False, "longmemeval_s"
            )
            self.assertEqual(rows[0]["response"], "ok")
            self.assertEqual(set(rows[0]), {
                "question", "answer", "category", "response", "answerable"
            })
            self.assertIn("The date of the question is 2025-01-03", rows[0]["question"])

    def test_fullcontext_resume_keeps_context_stats(self):
        convs = [types.SimpleNamespace(sessions=[types.SimpleNamespace(
            date_time="2025-01-01T00:00:00",
            messages=[types.SimpleNamespace(
                speaker="A", to_string=lambda text=text: text
            )],
        )]) for text in ("first", "second")]
        questions = [[types.SimpleNamespace(
            content="Q?", answer="ok", category=1, date=None, answerable=True
        )] for _ in range(2)]

        class Full:
            def __init__(self, fail=False):
                self.fail = fail
                self.seen = []

            def answer_question(self, question, bench, chunks):
                self.seen.append(chunks[0])
                if self.fail and "second" in chunks[0]:
                    raise RuntimeError("interrupted")
                return "ok", {"context_mode": "fullcontext", "removed_chunks": 0}

        with tempfile.TemporaryDirectory() as directory:
            data_path = os.path.join(directory, "dataset.json")
            output = os.path.join(directory, "answers.json")
            with open(data_path, "w") as handle:
                handle.write("[]")
            config = {"method": "fullcontext", "context_budget": 28000}
            self.runner.prepare_run(output, data_path, config, False, False)
            with patch.object(self.runner.ConvLoader, "load_locomo", return_value=(convs, questions)):
                with self.assertRaises(RuntimeError):
                    self.runner.process_all_conversations(
                        Full(fail=True), "fullcontext", data_path, output, "locomo", 1,
                        question_max_workers=1,
                    )
                self.runner.prepare_run(output, data_path, config, True, False)
                resumed = Full()
                self.runner.process_all_conversations(
                    resumed, "fullcontext", data_path, output, "locomo", 1,
                    question_max_workers=1, resume=True,
                )
            self.assertEqual(len(resumed.seen), 1)
            self.assertIn("second", resumed.seen[0])
            with open(output + ".context.json") as handle:
                self.assertEqual(set(json.load(handle)), {"0", "1"})


if __name__ == "__main__":
    unittest.main()
