"""Per-conversation timing and OpenAI-compatible API usage for benchmark runs."""

import copy
import json
import os
import statistics
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

from evaluation.protocol import atomic_json


_active = ContextVar("benchmark_measurement", default=None)


def _bucket():
    return {
        "wall_s": 0.0,
        "llm": {"calls": 0, "failed": 0, "missing_usage": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "request_wall_s": 0.0},
        "embedding": {"calls": 0, "failed": 0, "missing_usage": 0,
                      "prompt_tokens": 0, "request_wall_s": 0.0},
    }


class Measurement:
    def __init__(self, answer_path, answers=None, resume=False):
        self.path = answer_path + ".metrics.json"
        self.lock = threading.Lock()
        if resume and os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as handle:
                self.data = json.load(handle)
            if self.data.get("schema") != 1:
                raise ValueError("Cannot resume: measurement schema differs")
        elif resume and answers:
            raise ValueError("Cannot resume: measurements are missing")
        else:
            self.data = {"schema": 1, "conversations": {}, "judge_wall_s": None}
            atomic_json(self.path, self.data)
        for key, rows in (answers or {}).items():
            measured = self.data["conversations"].get(key)
            if measured is None or set(measured.get("questions", {})) != {str(i) for i in range(len(rows))}:
                raise ValueError(f"Cannot resume: incomplete measurements for conversation {key}")

    def _entry(self, conv, question, phase):
        item = self.data["conversations"].setdefault(str(conv),
                                                       {"construct": _bucket(), "wall_s": 0.0,
                                                        "questions": {}})
        if question is None:
            return item["construct"] if phase == "construct" else item
        query = item["questions"].setdefault(str(question),
                                               {"query": _bucket(), "retrieve": _bucket(), "answer": _bucket()})
        return query[phase]

    @contextmanager
    def phase(self, conv, question, phase):
        token = _active.set((self, conv, question, phase))
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self.lock:
                self._entry(conv, question, phase)["wall_s"] += elapsed
            _active.reset(token)

    def record(self, conv, question, phase, kind, response, elapsed, failed=False):
        usage = getattr(response, "usage", None) if response is not None else None
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None) if kind == "llm" else None
        with self.lock:
            item = self._entry(conv, question, phase)[kind]
            item["calls"] += 1
            item["failed"] += int(failed)
            item["request_wall_s"] += elapsed
            if not failed:
                if prompt is None or (kind == "llm" and completion is None):
                    item["missing_usage"] += 1
                else:
                    item["prompt_tokens"] += int(prompt)
                    if kind == "llm":
                        item["completion_tokens"] += int(completion)

    def save_conversation(self, conv):
        with self.lock:
            if str(conv) not in self.data["conversations"]:
                raise ValueError(f"No measurements for conversation {conv}")
            snapshot = copy.deepcopy(self.data)
        atomic_json(self.path, snapshot)

    def set_recmem_operations(self, conv, before, after):
        with self.lock:
            self.data["conversations"][str(conv)]["recmem_op_tokens"] = {
                name: {field: after["op_tokens"][name][field] - values[field]
                       for field in ("prompt_tokens", "completion_tokens", "total_tokens")}
                for name, values in before["op_tokens"].items()
            }

    def set_sizes(self, conv, input_units, memory_items=None):
        with self.lock:
            item = self.data["conversations"][str(conv)]
            item["input_units"] = input_units
            item["memory_items"] = memory_items

    def save_judge(self, elapsed):
        with self.lock:
            self.data["judge_wall_s"] = elapsed
            snapshot = copy.deepcopy(self.data)
        atomic_json(self.path, snapshot)


@contextmanager
def capture_openai():
    """Capture the sync SDK calls used by all six methods, without changing prompts."""
    from openai.resources.chat.completions import Completions
    from openai.resources.embeddings import Embeddings

    def wrap(original, kind):
        def measured(resource, *args, **kwargs):
            current = _active.get()
            if current is None:
                return original(resource, *args, **kwargs)
            meter, conv, question, phase = current
            start = time.perf_counter()
            try:
                result = original(resource, *args, **kwargs)
            except Exception:
                meter.record(conv, question, phase, kind, None, time.perf_counter() - start, failed=True)
                raise
            meter.record(conv, question, phase, kind, result, time.perf_counter() - start)
            return result
        return measured

    with patch.object(Completions, "create", wrap(Completions.create, "llm")), \
         patch.object(Embeddings, "create", wrap(Embeddings.create, "embedding")):
        yield


def summarize(path, expected):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    conversations = data.get("conversations", {})
    if data.get("schema") != 1 or set(conversations) != set(expected):
        raise ValueError(f"Incomplete measurement conversations: {path}")
    constructs, queries, retrieves, answers = [], [], [], []
    tokens = {phase: {"llm": {"prompt_tokens": 0, "completion_tokens": 0,
                              "missing_usage": 0, "failed": 0},
                      "embedding": {"prompt_tokens": 0, "missing_usage": 0, "failed": 0}}
              for phase in ("construct", "query")}
    calls = {"llm": {"calls": 0, "failed": 0, "missing_usage": 0,
                     "prompt_tokens": 0, "completion_tokens": 0, "request_wall_s": 0.0},
             "embedding": {"calls": 0, "failed": 0, "missing_usage": 0,
                           "prompt_tokens": 0, "request_wall_s": 0.0}}
    for key, count in expected.items():
        item = conversations[key]
        if set(item.get("questions", {})) != {str(i) for i in range(count)}:
            raise ValueError(f"Incomplete measurement questions: {path}, {key}")
        constructs.append(item["construct"]["wall_s"])
        for phase, bucket in [("construct", item["construct"])]:
            for kind in calls:
                for field in calls[kind]:
                    calls[kind][field] += bucket[kind][field]
                for field in tokens[phase][kind]:
                    tokens[phase][kind][field] += bucket[kind][field]
        for question in item["questions"].values():
            queries.append(question["query"]["wall_s"])
            retrieves.append(question["retrieve"]["wall_s"])
            answers.append(question["answer"]["wall_s"])
            for phase in ("retrieve", "answer"):
                bucket = question[phase]
                for kind in calls:
                    for field in calls[kind]:
                        calls[kind][field] += bucket[kind][field]
                    for field in tokens["query"][kind]:
                        tokens["query"][kind][field] += bucket[kind][field]

    def latency(values):
        return {"mean_s": statistics.mean(values), "p50_s": statistics.median(values),
                "p95_s": statistics.quantiles(values, n=100, method="inclusive")[94]
                if len(values) > 1 else values[0]}

    return {"conversations": len(constructs), "questions": len(queries),
            "input_units": sum(item.get("input_units", 0) for item in conversations.values()),
            "memory_items": [item.get("memory_items") for item in conversations.values()],
            "construct_llm_tokens_per_conversation":
                (tokens["construct"]["llm"]["prompt_tokens"] + tokens["construct"]["llm"]["completion_tokens"])
                / len(constructs) if not tokens["construct"]["llm"]["missing_usage"] else None,
            "query_llm_tokens_per_question":
                (tokens["query"]["llm"]["prompt_tokens"] + tokens["query"]["llm"]["completion_tokens"])
                / len(queries) if not tokens["query"]["llm"]["missing_usage"] else None,
            "tokens": tokens,
            "construct_latency": latency(constructs), "query_latency": latency(queries),
            "retrieve_latency": latency(retrieves), "answer_latency": latency(answers),
            "construct_wall_s": sum(constructs), "query_wall_s": sum(queries),
            "method_wall_s": sum(item["wall_s"] for item in conversations.values()),
            "judge_wall_s": data.get("judge_wall_s"), "calls": calls}
