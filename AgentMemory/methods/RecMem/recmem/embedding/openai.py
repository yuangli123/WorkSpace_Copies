import os
import re
from typing import List, Optional

from openai import OpenAI

import logging
import time

from recmem.embedding.base import Embedding

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_CONTEXT_ERROR = re.compile(
    r"maximum context length is (\d+) tokens.*?request has (\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


def _is_context_error(error: Exception) -> bool:
    return (
        getattr(error, "status_code", 400) == 400
        and "maximum context length" in str(error).lower()
    )


def _shorten_for_context_error(text: str, error: Exception) -> Optional[str]:
    """Use the server's token counts to shrink a rejected input offline."""
    if not _is_context_error(error):
        return None
    match = _CONTEXT_ERROR.search(str(error))
    if match:
        limit, actual = map(int, match.groups())
        keep = int(len(text) * max(1, limit - 64) / max(1, actual))
    else:
        keep = len(text) // 2
    keep = min(keep, len(text) - 1)
    if keep < 1:
        return None
    logger.warning("Embedding context overflow: shortening input from %d to %d characters", len(text), keep)
    return text[:keep]


class OpenAIEmbedding(Embedding):
    def __init__(
        self, model: Optional[str] = None, embedding_dims: Optional[int] = None
    ):

        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dims = embedding_dims or int(os.getenv("EMBEDDING_DIM", "1536"))
        self.send_dimensions = os.getenv(
            "EMBEDDING_SEND_DIMENSIONS", "true"
        ).lower() in {"1", "true", "yes"}

        api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = (
            os.getenv("EMBEDDING_API_BASE")
            or os.getenv("EMBEDDING_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def dim(self) -> int:
        return self.embedding_dims

    def embed(self, text: str) -> List[float]:
        """
        Get the embedding for the given text using OpenAI.

        Args:
            text (str): The text to embed.
        Returns:
            list: The embedding vector.
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        text = text.replace("\n", " ")
        last_error: Optional[Exception] = None
        for i in range(3):  # Retry 3 times
            try:
                request = {"input": [text], "model": self.model}
                if self.send_dimensions:
                    request["dimensions"] = self.embedding_dims
                embedding = self.client.embeddings.create(**request).data[0].embedding
                return embedding
            except Exception as e:
                last_error = e
                shorter = _shorten_for_context_error(text, e)
                if shorter is not None:
                    text = shorter
                    continue
                if getattr(e, "status_code", None) == 400:
                    raise
                logger.warning("Error embedding text: %s. Retrying (%d/3).", e, i + 1)
                if i < 2:
                    time.sleep(1 + i * 2)

        raise RuntimeError(
            f"Failed to embed text after 3 retries. Last error: {last_error}"
        ) from last_error

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single API call.

        Args:
            texts: List of texts to embed.
        Returns:
            List of embedding vectors in the same order as input texts.
        Raises:
            RuntimeError: If all retry attempts fail.
        """
        if not texts:
            return []

        # Preprocess texts
        processed_texts = [text.replace("\n", " ") for text in texts]
        last_error: Optional[Exception] = None

        for i in range(3):  # Retry 3 times
            try:
                request = {"input": processed_texts, "model": self.model}
                if self.send_dimensions:
                    request["dimensions"] = self.embedding_dims
                response = self.client.embeddings.create(**request)
                # Sort by index to ensure correct order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
            except Exception as e:
                last_error = e
                if _is_context_error(e):
                    # Keep short items intact; only the offending item is shortened by embed().
                    return [self.embed(text) for text in processed_texts]
                if getattr(e, "status_code", None) == 400:
                    raise
                logger.warning("Error embedding batch: %s. Retrying (%d/3).", e, i + 1)
                if i < 2:
                    time.sleep(1 + i * 2)

        raise RuntimeError(
            f"Failed to embed batch after 3 retries. Last error: {last_error}"
        ) from last_error
