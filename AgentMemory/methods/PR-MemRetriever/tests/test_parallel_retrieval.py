import json
from threading import Barrier, Event

import pytest

from evaluation.compare_parallel import compare
from evaluation.retrieval import evaluate
from experiments.io import read_jsonl, write_json
from memory.journal import ListDB
from memory.protocol import BATCH_STOP, Completion, parse_batch, validate_call
from methods.retrieval.tree import RetrievalEnvironment


SOURCE = "Alice moved to Paris.\nShe kept her old address for mail."
FACT = ("search_facts", {"query": "Alice city"})
RAW = ("search_turns", {"query": "Alice city"})
FINISH = ("finish", {"answer": "Paris"})


def completion(*calls, envelope=True):
    text = "".join("<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"
                   for name, args in calls)
    return "<tool_calls>" + text + BATCH_STOP if envelope else text


class MemoryDB(ListDB):
    def __init__(self):
        super().__init__()
        self.add("s_facts", ["fact"], ["Alice lives in Paris."], [{
            "memory_ref": "f1", "memory_content": "Alice lives in Paris.",
            "original_turns": [SOURCE], "source_turn_ids": ["s:1"],
        }])
        self.add("s_turns", ["raw"], [SOURCE], [{"memory_ref": "r1", "source_turn_ids": ["s:1"]}])
        self.order = []
        self.gate = None
        self.raw_completed = Event()

    def search(self, name, *args, **kwargs):
        result = super().search(name, *args, **kwargs)
        if self.gate is not None:
            self.gate.wait(timeout=5)
            if name == "s_facts" and not self.raw_completed.wait(timeout=5):
                raise TimeoutError("The other tool did not finish")
        self.order.append(name)
        if name == "s_turns":
            self.raw_completed.set()
        return result


def config(width=2, serial=False, steps=3):
    return {"model": {"max_new_tokens": 64}, "memory": {"retrieval_topk": 5}, "seed": 42,
            "retrieval": {"max_depth": steps, "max_parallel_calls": width, "serial_tool_execution": serial},
            "evaluation": {"max_steps": steps, "max_parallel_calls": width,
                           "serial_tool_execution": serial, "fallback_answer": False}}


def test_parallel_backend_reads_overlap_but_observations_are_ordered(tmp_path):
    db = MemoryDB()
    parallel = RetrievalEnvironment(db, "s", config(), tmp_path / "parallel")
    sequential = RetrievalEnvironment(db, "s", config(serial=True), tmp_path / "sequential")
    text = completion(FACT, RAW)
    slow_trace, fast_trace = {}, {}
    expected = sequential.step(text, [], trace=slow_trace)
    assert db.order == ["s_facts", "s_turns"]
    db.order.clear()
    db.raw_completed.clear()
    db.gate = Barrier(2)
    actual = parallel.step(text, [], trace=fast_trace)
    assert db.order == ["s_turns", "s_facts"]
    assert actual == expected
    assert actual[3].count(SOURCE) == 1
    assert actual[4][0]["memory_ref"] == "f1"
    assert actual[5] == [SOURCE]
    assert fast_trace["result_refs"] == slow_trace["result_refs"] == [["f1"], ["r1"]]


@pytest.mark.parametrize("text,remaining,error", [
    (completion(FACT, FINISH), 2, "format"),
    (completion(FACT, RAW)[:-2], 2, "format"),
    (completion(FACT, RAW, FACT), 4, "format"),
    (completion(FACT, RAW, envelope=False), 2, "format"),
    (completion(FACT, ("unknown_tool", {})), 2, "format"),
    (completion(FACT, ("search_turns", {"query": "Alice", "top_k": True})), 2, "format"),
    (completion(FACT, RAW), 1, "search_budget"),
])
def test_invalid_batch_never_executes_valid_prefix(tmp_path, text, remaining, error):
    db = MemoryDB()
    env = RetrievalEnvironment(db, "s", config(), tmp_path)
    stats = {}
    valid, terminal, _, _, _, _ = env.step(text, [], remaining_calls=remaining, trace=stats)
    assert not valid and terminal and db.order == []
    assert stats["error"] == error and stats["search_calls"] == 0


def test_legacy_single_call_and_prompt_are_kept(tmp_path):
    db = MemoryDB()
    baseline = RetrievalEnvironment(db, "s", config(width=1), tmp_path / "original")
    parallel = RetrievalEnvironment(db, "s", config(), tmp_path / "parallel")
    assert baseline.prompt("Where?", 1) == baseline.retriever.construct_prompt("Where?", "1")
    parallel_prompt = parallel.prompt("Where?", 1)[0]["content"]
    assert parallel_prompt.startswith(baseline.prompt("Where?", 1)[0]["content"])
    example_header = "[Two-call example]"
    assert parallel_prompt.count(example_header) == 1
    example = parallel_prompt.split(example_header, 1)[1].strip()
    example_calls = parse_batch(example)
    assert [call["name"] for call in example_calls] == ["search_facts", "search_turns"]
    assert all(validate_call(call, parallel.retriever.tool_map[call["name"]]) for call in example_calls)
    text = completion(FACT, envelope=False)
    assert baseline.step(text, []) == parallel.step(text, [])
    assert not baseline.step(completion(FACT, RAW), [])[0]
    assert parallel.step(completion(FINISH), [], remaining_calls=0, trace={})[:3] == (True, True, "Paris")


def test_all_existing_tool_formatters_are_reused(tmp_path):
    db = MemoryDB()
    for kind in ("experiences", "personas", "summary"):
        db.add("s_" + kind, ["Alice"], ["Example"], [{
            "memory_ref": kind, "memory_content": "Example", "original_turns": [SOURCE],
        }])
    env = RetrievalEnvironment(db, "s", config(), tmp_path)
    cases = [FACT, RAW, ("search_experiences", {"query": "Alice"}),
             ("search_personas", {"query": "", "name": "Alice"}),
             ("search_personas", {"query": "Alice"}),
             ("search_summary", {"query": ""}), ("search_summary", {"query": "Alice"})]
    for name, args in cases:
        tool = env.retriever.tool_map[name]
        original_seen, deferred_seen = {SOURCE}, {SOURCE}
        original, original_items = tool(sample_id="s", seen_turns=original_seen, **args)
        render, items = tool(sample_id="s", defer_format=True, **args)
        assert render(deferred_seen) == original
        assert items == original_items and deferred_seen == original_seen


def test_width_eight_uses_a_separate_search_budget_from_decision_depth(tmp_path):
    from methods.retrieval.tree import sample_tree
    import random

    class Policy:
        def complete(self, messages, **options):
            if len(messages) == 2:
                assert "Generate at most 8 search calls" in messages[-1]["content"]
                text = completion(*([FACT, RAW] * 4))
            else:
                assert "Search calls remaining: 0" in messages[-1]["content"]
                text = completion(FINISH)
            return Completion(text, [1], [2])

    cfg = config(width=8, steps=3)
    cfg["retrieval"].update(expansion_rounds=0, max_search_calls=8)
    env = RetrievalEnvironment(MemoryDB(), "s", cfg, tmp_path)
    assert "one to 8" in env.prompt("Where?", 1)[0]["content"]
    nodes = sample_tree(Policy(), env, {"question": "Where?", "category": 1}, cfg, random.Random(42))
    assert nodes[1].evaluation["search_calls"] == 8
    assert nodes[-1].answer == "Paris" and nodes[-1].depth == 2


def test_scripted_pair_uses_real_evaluation_entrypoint(tmp_path):
    qa = {"sample_id": "s", "qa_id": "1", "question": "Where does Alice live?", "answer": "Paris", "category": 1}

    class Memory:
        questions = [qa]

        def environment(self, qa, cfg, log_path):
            return RetrievalEnvironment(MemoryDB(), qa["sample_id"], cfg, log_path)

    class Policy:
        def __init__(self, parallel):
            self.parallel = parallel
            self.requests = 0

        def complete(self, messages, **options):
            self.requests += 1
            assert options.get("tool_stop") == (BATCH_STOP if self.parallel else None)
            if self.requests == 1:
                text = completion(FACT, RAW) if self.parallel else completion(FACT, envelope=False)
            elif self.requests == 2 and not self.parallel:
                text = completion(RAW, envelope=False)
            else:
                assert SOURCE in "".join(m["content"] for m in messages)
                if self.parallel:
                    assert "Search calls remaining: 1." in messages[-1]["content"]
                text = completion(FINISH, envelope=self.parallel)
            return Completion(text, [1], [2])

        def check_observation(self, text):
            pass

    for name, width in (("baseline", 1), ("parallel", 2)):
        cfg = config(width=width)
        out = tmp_path / name
        result = evaluate(cfg, Memory(), Policy(width == 2), out / "answers.jsonl")
        assert result["f1"] == result["format_success_rate"] == 1
        assert result["mean_search_calls"] == 2
        assert result["mean_decisions"] == (2 if width == 2 else 3)
        assert result["latency_p95_seconds"] >= 0
        write_json(out / "manifest.json", {"config": {"evaluation": cfg["evaluation"]},
                                            "inputs": {"memory": "fixed"}, "code_sha256": {}})

    report = compare(tmp_path / "baseline", tmp_path / "parallel")
    assert report["delta"]["f1"] == 0
    assert report["delta"]["mean_decisions"] == -1
    parallel_row = read_jsonl(tmp_path / "parallel" / "answers.jsonl")[0]
    assert parallel_row["traces"][0]["result_refs"] == [["f1"], ["r1"]]
