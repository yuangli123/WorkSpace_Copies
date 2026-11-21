from evaluation.guided_parallel import GuidedPolicy, batch_grammar, round_diagnostics
from memory.protocol import Completion


def prior(width):
    inner = '<tool_call>{"name":"search_facts","arguments":{"query":"before"}}</tool_call>'
    return {"role": "assistant", "content": "<tool_calls>" + inner * width + "</tool_calls>"}


class RecordingPolicy:
    def complete(self, messages, **kwargs):
        self.messages = messages
        self.options = kwargs
        return Completion("", [1], [2])


def test_two_parallel_rounds_use_the_observation_then_finish():
    policy = RecordingPolicy()
    guided = GuidedPolicy(policy, "fixed-parallel")
    observation = {"role": "user", "content": "<observation>Missing the project date</observation>"}
    messages = [{"role": "user", "content": "Question"}, prior(2), observation]
    guided.complete(messages, temperature=0)
    assert policy.options["guided_grammar"] == batch_grammar((2,), allow_finish=False)
    assert policy.messages[2]["content"].startswith(observation["content"])
    assert messages[2] == observation
    guided.complete(messages + [prior(2), observation])
    assert policy.options["guided_grammar"] == batch_grammar((), allow_finish=True)


def test_adaptive_can_finish_after_first_batch_and_uses_remaining_budget():
    policy = RecordingPolicy()
    guided = GuidedPolicy(policy, "adaptive-parallel")
    question = {"role": "user", "content": "Question"}
    guided.complete([question])
    assert policy.options["guided_grammar"] == batch_grammar((2,), allow_finish=False)
    guided.complete([question, prior(2), question])
    assert policy.options["guided_grammar"] == batch_grammar((1, 2), allow_finish=True)
    guided.complete([question, prior(2), question, prior(1), question])
    assert policy.options["guided_grammar"] == batch_grammar((1,), allow_finish=True)
    guided.complete([question], stop_at_tool=False)
    assert policy.options == {"stop_at_tool": False}


def test_width_sweep_uses_the_selected_width_and_at_most_two_search_rounds():
    policy = RecordingPolicy()
    for width in (2, 4, 8):
        guided = GuidedPolicy(policy, "width-sweep", width=width)
        query = {"role": "user", "content": "Question"}
        guided.complete([query])
        assert policy.options["guided_grammar"] == batch_grammar((width,), allow_finish=False)
        guided.complete([query, prior(width), query])
        assert policy.options["guided_grammar"] == batch_grammar(tuple(range(1, width + 1)), allow_finish=True)
        guided.complete([query, prior(width), query, prior(1), query])
        assert policy.options["guided_grammar"] == batch_grammar((), allow_finish=True)


def test_round_diagnostics_counts_novel_followup_calls():
    calls = [{"name": "search_facts", "arguments": {"query": q}} for q in ("a", "b", "c")]
    rows = [{"traces": [{"search_calls": 2, "calls": calls[:2]},
                        {"search_calls": 2, "calls": calls[1:]}]}]
    result = round_diagnostics(rows)
    assert result["questions_with_at_least_two_parallel_rounds"] == 1
    assert result["later_parallel_rounds_with_a_new_call"] == 1
    assert result["repeated_search_calls"] == 1
