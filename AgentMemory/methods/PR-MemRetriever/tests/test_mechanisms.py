from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch

from methods.construction.credit import hindsight_scores, select_operations
from memory.data import prepare_locomo
from experiments.io import load_config, read_json, read_jsonl, write_jsonl
from methods.retrieval.loss import mot_loss
from memory.journal import JournalDB, ListDB
from memory.protocol import Completion, parse_calls
from methods.retrieval.tree import Node, answer_f1, score_forest, training_rows


def test_distributed_scheduler_zero_warmup_updates_first_step():
    pytest.importorskip("tensordict")
    from training.verl_backend import enable_verl
    enable_verl()
    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=0)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1)
    parameter.sum().backward()
    optimizer.step()
    scheduler.step()
    assert parameter.item() == pytest.approx(0.9)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1)


def test_distributed_scheduler_preserves_positive_warmup():
    pytest.importorskip("tensordict")
    from training.verl_backend import enable_verl
    enable_verl()
    from verl.utils.torch_functional import get_constant_schedule_with_warmup
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2)
    assert optimizer.param_groups[0]['lr'] == 0
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.05)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]['lr'] == pytest.approx(0.1)


def test_export_rejects_incomplete_checkpoint(tmp_path):
    from training.verl_backend import export_checkpoint
    checkpoint, output = tmp_path / "checkpoint", tmp_path / "export"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="Incomplete"):
        export_checkpoint(checkpoint, output)
    (checkpoint / "COMPLETE").write_text("ok\n")
    (checkpoint / "state.json").write_text(json.dumps({"step": 2, "world_size": 4}))
    with pytest.raises(ValueError, match="Missing model shard for rank 0"):
        export_checkpoint(checkpoint, output)
    assert not output.exists()


@pytest.fixture
def config():
    cfg = load_config("configs/memt4b.yaml")
    cfg["retrieval"].update(expansion_rounds=1, evidence_alpha=1.0, trees=2, max_depth=3, pivots=2,
                            steps=2, batch_size=1, save_every=1, max_prompt_tokens=16000, max_observation_tokens=16000)
    cfg["memory"]["backend"] = "lexical_test"
    cfg["construction"].update(steps=2, deepspeed=None, save_every=1, per_device_batch_size=1, gradient_accumulation=1)
    cfg["model"].update(device="cpu", dtype="float32", max_new_tokens=32)
    return cfg


def test_child_mean_dual_advantages(config):
    # Unequal branch sizes: mean-of-children=0.5, mean-of-descendant-leaves=1/3.
    tree = [Node(0, -1, 0, []), Node(1, 0, 1, [], children=[2, 3]),
            Node(2, 1, 2, [], terminal=True, answer="Paris"), Node(3, 1, 2, [], children=[4, 5]),
            Node(4, 3, 3, [], terminal=True, answer="Rome"), Node(5, 3, 3, [], terminal=True, answer="Rome")]
    other = [Node(0, -1, 0, []), Node(1, 0, 1, [], terminal=True, answer="Rome", valid=False)]
    forest = score_forest([tree, other], {"answer": "Paris", "evidence": []}, config)
    assert tree[1].performance == 0.5
    assert other[1].reward == 0
    assert abs(sum(n.intra for n in tree[1:])) < 1e-5
    assert abs(sum(n.inter for t in forest for n in t[1:])) < 1e-5
    assert tree[2].advantage > tree[4].advantage


def test_evidence_mask_and_constant_rewards(config):
    node = Node(1, 0, 1, [], valid=False, terminal=True, answer="Paris",
                retrieved=[{"source_turn_ids": [["D1:1"], ["D1:2"]]}])
    score_forest([[Node(0, -1, 0, []), node]], {"answer": "Paris", "evidence": ["D1:1"]}, config)
    assert node.evidence == 1 and node.reward == 0 and node.advantage == 0


def test_loss_masks_node_weights_and_negative_advantage():
    current = torch.tensor([[0.0, -1.0], [-1.0, -2.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    old = current.detach().clone()
    loss, _ = mot_loss(current, old, old, torch.tensor([1.0, -1.0]), mask,
                       torch.tensor([0.25, 0.75]), beta=0)
    assert loss.item() == pytest.approx(0.5)
    loss.backward()
    assert current.grad[0, 1] == 0
    assert current.grad[0, 0] < 0 and current.grad[1, 0] > 0


def test_provenance_update_delete_and_snapshot(tmp_path):
    db = JournalDB(ListDB())
    db.create_collection("s_facts")
    first = {"op_id": "add", "memory_refs": [], "deleted_refs": [], "parent_operation_ids": ["form"]}
    db.begin(first)
    db.add("s_facts", ["m"], ["Paris"], [{"source_turn_ids": ["D1:1"]}])
    db.end()
    update = {"op_id": "update", "memory_refs": [], "deleted_refs": []}
    db.begin(update)
    db.upsert("s_facts", ["m"], ["Rome"], [{"source_turn_ids": ["D2:1"]}])
    meta = db.get("s_facts")["metadatas"][0]
    assert meta["memory_version"] == 2
    assert set(meta["operation_ids"]) == {"add", "form", "update"}
    assert set(meta["source_turn_ids"]) == {"D1:1", "D2:1"}
    delete = {"op_id": "delete", "memory_refs": [], "deleted_refs": []}
    db.begin(delete)
    db.delete("s_facts", ["m"])
    db.end()
    db.save(tmp_path / "snapshot.json", ["s_facts"])
    restored = JournalDB.restore(tmp_path / "snapshot.json", ListDB())
    assert not restored.get("s_facts")["ids"]
    assert len(restored.versions) == 2 and delete["deleted_refs"] == update["memory_refs"]
    assert restored.versions[first["memory_refs"][0]]["document"] == "Paris"


def test_hindsight_two_gates_negative_scores_and_ties(config):
    op = {"op_id": "o", "sample_id": "s", "source_turn_ids": ["D1:1"], "valid": True,
          "phase": "evolution", "category": "ignore_item"}
    qa = {"qa_id": "q", "sample_id": "s", "evidence": ["D1:1"]}
    root = asdict(Node(0, -1, 0, []))
    leaf = asdict(Node(1, 0, 1, [], terminal=True, advantage=-2,
                       retrieved=[{"operation_ids": ["o"]}]))
    record = {"qa_id": "q", "sample_id": "s", "trees": [[root, leaf]]}
    scored = hindsight_scores([op], [record], [qa], config)
    assert scored[0]["score"] == pytest.approx(-2.2)
    config["construction"]["evidence_gate"] = False
    assert hindsight_scores([op], [record], [qa], config)[0]["score"] == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="Duplicate QA"):
        hindsight_scores([op], [record, record], [qa], config)
    with pytest.raises(ValueError, match="different episode"):
        hindsight_scores([op], [dict(record, sample_id="another")], [qa], config)
    tied = [dict(scored[0], op_id=str(i), score=0) for i in range(4)]
    selected, audit = select_operations(tied)
    assert len(selected) == 2 and audit["evolution:ignore_item"]["distinct_scores"] == 1


def test_split_is_conversation_order_and_excludes_adversarial(tmp_path):
    data = [{"sample_id": str(i), "conversation": {"session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "Paris"}]},
             "qa": [{"question": "Where?", "answer": "Paris", "evidence": ["D1:1"], "category": category} for category in (1, 5)]} for i in range(10)]
    source = tmp_path / "locomo.json"
    source.write_text(json.dumps(data))
    prepare_locomo(source, tmp_path / "prepared")
    split = read_json(tmp_path / "prepared/split.json")["ids"]
    assert split == {"train": ["0"], "valid": ["1"], "test": [str(i) for i in range(2, 10)]}
    assert len(read_jsonl(tmp_path / "prepared/test/questions.jsonl")) == 8
    episodes = read_jsonl(tmp_path / "prepared/train/episodes.jsonl")
    assert "answer" not in json.dumps(episodes) and "evidence" not in json.dumps(episodes)


def test_protocol_rejects_partial_json():
    assert not parse_calls('<tool_call>{"name":"finish","arguments":{"answer":"Paris"}}</tool_call><tool_call>bad')
    assert answer_f1("The Paris!", "Paris") == 1


class ScriptedPolicy:
    """A test fixture only: deterministic memory/tool decisions, never an experiment model."""
    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kwargs):
        prompt = messages[0]["content"]
        if "Memory System Analyzer" in prompt:
            call = {"name": "create_fact", "arguments": {"fact": "Alice lives in Paris", "start_time": "", "end_time": ""}}
        elif "Insights Database Administrator" in prompt:
            call = {"name": "add_item", "arguments": {"document": "Alice lives in Paris"}}
        elif len(messages) == 2:
            call = {"name": "search_facts", "arguments": {"query": "Alice city"}}
        else:
            self.calls += 1
            call = {"name": "finish", "arguments": {"answer": "Paris" if self.calls % 2 else "Rome"}}
        text = "<tool_call>" + json.dumps(call) + "</tool_call>"
        return Completion(text, [1, 2], [3, 4])

    def check_observation(self, text):
        pass


def test_construction_tree_hindsight_closed_loop(config, tmp_path):
    from experiments.workflow import MemoryRun, build, collect_forests, derive_sft
    from experiments.io import write_json
    episode = {"sample_id": "sample", "benchmark": "locomo", "sessions": [
        {"session_id": "session1", "time": "2023-01-01", "turns": [{"turn_id": "D1:1", "speaker": "Alice", "text": "I live in Paris"}]}]}
    qa = {"qa_id": "q", "sample_id": "sample", "question": "Where does Alice live?", "answer": "Paris", "evidence": ["D1:1"], "category": 1}
    source = tmp_path / "data"
    write_jsonl(source / "episodes.jsonl", [episode])
    write_jsonl(source / "questions.jsonl", [qa])
    write_json(source / "partition.json", {"split": "train"})
    policy = ScriptedPolicy()
    build(config, source, tmp_path / "memory", policy)
    memory = MemoryRun(tmp_path / "memory", config)
    assert len(memory.operations) == 2
    collect_forests(config, memory, policy, tmp_path / "forests")
    derive_sft(config, memory, tmp_path / "forests/forests.jsonl", tmp_path / "sft")
    selected = read_jsonl(tmp_path / "sft/selected.jsonl")
    assert {s["phase"] for s in selected} == {"formation", "evolution"}
    assert all(s["trace_hits"] > 0 for s in selected)
    record = read_jsonl(tmp_path / "forests/forests.jsonl")[0]
    assert any(n["advantage"] != 0 for t in record["trees"] for n in t[1:])
    rows = training_rows([[Node(**n) for n in t] for t in record["trees"]])
    assert sum(r["weight"] for r in rows) == pytest.approx(1)
    assert len({(r["tree"], r["node"]) for r in rows}) == len(rows)


def test_test_partition_cannot_train(config, tmp_path):
    from experiments.workflow import derive_sft
    class Memory:
        partition = {"split": "test"}
    with pytest.raises(ValueError, match="train partition"):
        derive_sft(config, Memory(), "unused", tmp_path)


def test_chroma_adapter_does_not_invent_memory_and_does_not_swallow_errors():
    from memory.vector_db import ChromaVectorDB
    class Collection:
        def __init__(self):
            self.size = 0
        def count(self):
            return self.size
        def get(self, **kwargs):
            return {"ids": [], "documents": [], "metadatas": []}
        def add(self, **kwargs):
            raise AssertionError("Reads must not insert dummy memories")
        def query(self, **kwargs):
            raise RuntimeError("retriever failed")
    collection = Collection()
    class Client:
        def get_collection(self, **kwargs):
            assert kwargs["embedding_function"] == "bge"
            return collection
    db = ChromaVectorDB.__new__(ChromaVectorDB)
    db.client, db.default_embedding_function = Client(), "bge"
    assert db.get("sample_facts")["ids"] == []
    assert db.search("sample_facts", query_texts=["city"])["ids"] == [[]]
    collection.size = 1
    with pytest.raises(RuntimeError, match="retriever failed"):
        db.search("sample_facts", query_texts=["city"])


def test_all_evolution_actions_remain_auditable(config):
    from methods.construction.rollout import construct_episode
    from memory.protocol import Completion
    config["memory"]["summary_context_turns"] = 1
    db = JournalDB(ListDB())
    class Policy:
        def __init__(self):
            self.action = 0
        def complete(self, messages, **kwargs):
            if "Memory System Analyzer" in messages[0]["content"]:
                call = {"name": "create_fact", "arguments": {"fact": "Alice city", "start_time": "", "end_time": ""}}
            else:
                names = ["add_item", "update_item", "ignore_item", "delete_item"]
                name = names[self.action]
                self.action += 1
                args = {"document": "Alice in Paris"} if name == "add_item" else {"reason": "same"} if name == "ignore_item" else {"id": db.get("sample_facts")["ids"][0]}
                if name == "update_item":
                    args["document"] = "Alice in Rome"
                call = {"name": name, "arguments": args}
            text = "<tool_call>" + json.dumps(call) + "</tool_call>"
            return Completion(text, [1], [2])
    episode = {"sample_id": "sample", "sessions": [{"session_id": "s1", "time": "2023-01-01", "turns": [
        {"turn_id": f"D1:{i}", "speaker": "Alice", "text": f"city {i}"} for i in range(4)]}]}
    operations, _ = construct_episode(episode, Policy(), db, config)
    evolution = [op for op in operations if op["phase"] == "evolution"]
    assert [op["category"] for op in evolution] == ["add_item", "update_item", "ignore_item", "delete_item"]
    assert all(op["valid"] for op in evolution)
    assert len(db.versions) == 6  # two factual versions plus four raw records
    assert not db.get("sample_facts")["ids"]
    assert evolution[2]["memory_refs"] == [] and evolution[3]["deleted_refs"]


def test_batched_forests_keep_independent_paths_and_seeds(config):
    from methods.retrieval.tree import sample_forests
    class Environment:
        def __init__(self, name):
            self.name = name
        def prompt(self, question, category):
            return [{"role": "user", "content": question}]
        def step(self, text, seen):
            if text == "search":
                return True, False, "", self.name, [{"memory_ref": self.name, "source_turn_ids": [self.name]}], [self.name]
            return True, True, text, "", [], seen
    class Policy:
        def __init__(self):
            self.requests = []
        def complete_batch(self, messages, seeds, **kwargs):
            self.requests.extend(seeds)
            return [Completion("search" if len(m) == 1 else ("Paris" if seed % 2 else "Rome"), [1], [2]) for m, seed in zip(messages, seeds)]
    questions = [{"question": "first", "category": 1, "answer": "Paris", "evidence": ["a"]},
                 {"question": "second", "category": 1, "answer": "Rome", "evidence": ["b"]}]
    policy = Policy()
    forests = sample_forests(policy, [Environment("a"), Environment("b")], questions, config, [41, 42])
    assert len(forests) == 2 and all(len(forest) == 2 for forest in forests)
    assert len(policy.requests) == len(set(policy.requests))
    for forest, key in zip(forests, ["a", "b"]):
        assert all(m["memory_ref"] == key for tree in forest for n in tree for m in n.retrieved)
        assert sum(r["weight"] for r in training_rows(forest)) == pytest.approx(1)
    again = sample_forests(Policy(), [Environment("a"), Environment("b")], questions, config, [41, 42])
    assert [[asdict(n) for tree in f for n in tree] for f in forests] == [[asdict(n) for tree in f for n in tree] for f in again]


def test_locomo_compound_and_missing_evidence_remain_auditable(tmp_path):
    from memory.data import locomo
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps([{"sample_id": "s", "conversation": {
        "session_1": [{"dia_id": "D1:1", "speaker": "A"}, {"dia_id": "D1:2", "speaker": "B"}]},
        "qa": [{"question": "q", "answer": "a", "category": 1, "evidence": ["D1:1; D1:2"]},
               {"question": "q2", "answer": "a2", "category": 1, "evidence": ["D1:1 D1:2"]},
               {"question": "q3", "answer": "a3", "category": 1, "evidence": ["D1:1", "D9:9"]}]}]))
    _, questions = locomo(source)
    assert len(questions) == 3
    assert questions[0]["evidence"] == questions[1]["evidence"] == ["D1:1", "D1:2"]
    assert questions[2]["evidence"] == []
    assert questions[2]["invalid_evidence"] == ["D9:9"]
    assert questions[2]["raw_evidence"] == ["D1:1", "D9:9"]
