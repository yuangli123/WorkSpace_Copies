"""Simple text baselines using the same conversation chunks as RecMem."""

from collections import deque
from typing import Dict, List, Tuple

from recmem.llm import OpenAIClient
from recmem.llm.base import Message


_INSTRUCTIONS = """Answer the question using the timestamped conversation below.
Use explicit evidence for factual questions and the speakers' stated preferences for advice.
For conflicting information, prefer the latest update. Interpret relative dates using
the message timestamp. If the conversation contains no answer, say so briefly.
{length_rule}

Conversation:
{context}
Question: {question}
Answer:"""


def _is_context_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(word in message for word in ("context", "maximum", "max seq len", "max_model_len")) and any(
        word in message for word in ("token", "length", "seq len")
    )


def _estimate_tokens(text: str) -> int:
    # ponytail: UTF-8 length is a deterministic offline estimate; use the model
    # tokenizer if an exact local tokenizer becomes available in the experiment env.
    return (len(text.encode("utf-8")) + 2) // 3


class ContextAnswerer:
    """Render answer prompts, optionally shrinking context to fit the service."""

    def __init__(self, llm: OpenAIClient, model: str, budget: int):
        if budget <= 0:
            raise ValueError("context_budget must be positive")
        self.llm = llm
        self.model = model
        self.budget = budget

    def answer(
        self, chunks: List[str], question: str, bench: str, *, newest_first: bool,
        truncate: bool = True,
    ) -> Tuple[str, Dict[str, object]]:
        length_rule = (
            "Answer in fewer than 5-6 words when possible."
            if bench == "locomo"
            else "Give a concise, direct answer."
        )
        selected = deque(chunks)
        lengths = deque(_estimate_tokens(chunk) for chunk in chunks)
        fixed_tokens = _estimate_tokens(
            _INSTRUCTIONS.format(length_rule=length_rule, context="", question=question)
        )
        context_tokens = sum(lengths)
        dropped = 0
        trimmed_tokens = 0
        removed_tokens = 0

        def render() -> str:
            return _INSTRUCTIONS.format(
                length_rule=length_rule,
                context="\n".join(selected) if selected else "None",
                question=question,
            )

        def shrink() -> bool:
            nonlocal dropped, trimmed_tokens, removed_tokens, context_tokens
            if len(selected) > 1:
                if newest_first:
                    selected.pop()
                    size = lengths.pop()
                else:
                    selected.popleft()
                    size = lengths.popleft()
                context_tokens -= size
                removed_tokens += size
                dropped += 1
                return True
            if selected:
                chunk = selected[0]
                if len(chunk) > 1:
                    keep = len(chunk) // 2
                    selected[0] = chunk[:keep] if newest_first else chunk[-keep:]
                    difference = _estimate_tokens(chunk) - _estimate_tokens(selected[0])
                    trimmed_tokens += difference
                    removed_tokens += difference
                    context_tokens -= lengths[0]
                    lengths[0] = _estimate_tokens(selected[0])
                    context_tokens += lengths[0]
                    return True
                selected.clear()
                lengths.clear()
                context_tokens = 0
                removed_tokens += _estimate_tokens(chunk)
                dropped += 1
                return True
            return False

        # The UTF-8 heuristic is approximate; server context errors provide the final guard.
        while truncate and fixed_tokens + context_tokens + len(selected) > self.budget:
            if not shrink():
                raise ValueError("context_budget is smaller than the answer prompt")
        while truncate and _estimate_tokens(render()) > self.budget:
            if not shrink():
                raise ValueError("context_budget is smaller than the answer prompt")

        while True:
            try:
                response = self.llm.chat_completion(
                    model=self.model,
                    messages=[Message(role="system", content=render())],
                    temperature=0.0,
                    max_tokens=512,
                )
                context_mode = "naive_rag" if newest_first else (
                    "windowed_fullcontext" if removed_tokens else "fullcontext"
                )
                return response.content.strip(), {
                    "context_mode": context_mode,
                    "removed_chunks": dropped,
                    "removed_tokens_estimate": removed_tokens,
                    "trimmed_tokens_estimate": trimmed_tokens,
                }
            except Exception as error:
                if not truncate or not _is_context_error(error) or not shrink():
                    raise


from methods.fullcontext import FullContext  # noqa: E402 - preserve existing import path
from methods.naive_rag import NaiveRAG  # noqa: E402 - preserve existing import path
