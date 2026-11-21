from evaluation.retrieval import locomo_abstention, metric_summary


def test_locomo_adversarial_accuracy_does_not_contaminate_memt_f1():
    row = {"f1": 0.5, "bleu1": 0.3, "format_errors": 0, "search_budget_errors": 0, "steps": 2,
           "search_rounds": 1, "parallel_rounds": 1, "finished": True, "fallback_answer": False,
           "search_calls": 2, "generated_tokens": 60, "prompt_tokens": 100,
           "policy_seconds": 1.0, "tool_seconds": 0.5, "latency_seconds": 1.5}
    adversarial = {**row, "f1": None, "bleu1": None, "abstention_correct": locomo_abstention("Not mentioned.")}
    assert not locomo_abstention("Went to Paris")
    assert locomo_abstention("No information available")
    summary = metric_summary([row, adversarial])
    assert summary["count"] == 2
    assert summary["f1_count"] == summary["abstention_count"] == 1
    assert summary["f1"] == 0.5 and summary["bleu1"] == 0.3
    assert summary["abstention_accuracy"] == 1.0
