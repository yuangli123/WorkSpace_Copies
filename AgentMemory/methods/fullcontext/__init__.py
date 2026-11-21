"""Full-context baseline that sends the complete conversation history."""

from typing import Dict, List, Tuple


class FullContext:
    def __init__(self, answerer):
        self.answerer = answerer

    def answer_question(
        self, question: str, bench: str, chunks: List[str]
    ) -> Tuple[str, Dict[str, object]]:
        return self.answerer.answer(chunks, question, bench, newest_first=False, truncate=False)
