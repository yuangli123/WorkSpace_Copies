import hashlib
import json
from types import SimpleNamespace

from methods.construction.rollout import operation


def completion(seed=None, protocol="paper-v2"):
    return SimpleNamespace(text='<tool_call>{"name":"ignore_item","arguments":{}}</tool_call>',
                           sampling_seed=seed, protocol_version=protocol)


def row(value, index=0):
    return operation("s", "evolution", value, [{"role": "user", "content": "same"}],
                     ["D1:1"], {"name": "ignore_item", "arguments": {}}, index, True)


def test_independent_identical_calls_have_distinct_ids():
    a, b = row(completion(11)), row(completion(22))
    assert a["op_id"] != b["op_id"]
    assert a["completion_id"] != b["completion_id"]
    assert row(completion(11)) == a


def test_actions_in_one_invocation_keep_completion_group():
    value = completion(11)
    value.text *= 2
    a, b = row(value, 0), row(value, 1)
    assert a["op_id"] != b["op_id"]
    assert a["completion_id"] == b["completion_id"]


def test_legacy_identity_unchanged():
    value = completion(None, "legacy-v1")
    a = row(value)
    identity = json.dumps(["s", "evolution", a["messages"], ["D1:1"], value.text, 0],
                          sort_keys=True, ensure_ascii=False)
    assert a["op_id"] == hashlib.sha256(identity.encode()).hexdigest()[:32]

