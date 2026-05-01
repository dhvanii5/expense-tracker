from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "").rstrip("/")
_LLAMA_SERVER_EMBEDDINGS_URL = os.getenv(
    "LLAMA_SERVER_EMBEDDINGS_URL",
    f"{_LLAMA_SERVER_URL}/embedding" if _LLAMA_SERVER_URL else "",
).strip()

_model_instance: Optional[object] = None
_load_attempted = False


class _LlamaServerEmbedAdapter:
    """Call llama-server's embedding endpoint and expose encode()."""

    def __init__(self, embeddings_url: str):
        self._embeddings_url = embeddings_url

    def encode(self, text: str) -> "_NumpyLike":
        payload = json.dumps({"input": text}).encode("utf-8")
        request = urllib.request.Request(
            self._embeddings_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Failed to reach llama-server embeddings API: {exc}") from exc

        embedding = None
        if isinstance(body, list) and body:
            embedding = body[0].get("embedding")
        elif isinstance(body, dict):
            embedding = body.get("embedding")
            if embedding is None:
                data = body.get("data")
                if isinstance(data, list) and data:
                    embedding = data[0].get("embedding")

        if not isinstance(embedding, list):
            raise RuntimeError("llama-server embeddings response did not include an embedding")

        # If llama-server returned token-level embeddings (2D array), perform mean pooling
        if len(embedding) > 0 and isinstance(embedding[0], list):
            num_tokens = len(embedding)
            pooled = []
            for i in range(len(embedding[0])):
                val = sum(token_emb[i] for token_emb in embedding) / num_tokens
                pooled.append(val)
            embedding = pooled

        # L2 normalize the vector to ensure accurate cosine similarity in ChromaDB
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return _NumpyLike(embedding)


class _NumpyLike:
    """Minimal shim so callers can keep using .tolist()."""

    def __init__(self, data: list[float]):
        self._data = data

    def tolist(self) -> list[float]:
        return self._data

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def load_embed_model() -> Optional[object]:
    """Load the embedding model once and cache it.

    Requires LLAMA_SERVER_EMBEDDINGS_URL (or LLAMA_SERVER_URL) to be set.
    Returns None when no embeddings URL is configured.
    """
    global _model_instance, _load_attempted

    if _model_instance is not None:
        return _model_instance

    if _load_attempted:
        return None

    _load_attempted = True

    if _LLAMA_SERVER_EMBEDDINGS_URL:
        logger.info(
            "Using llama-server embeddings API: %s", _LLAMA_SERVER_EMBEDDINGS_URL
        )
        _model_instance = _LlamaServerEmbedAdapter(_LLAMA_SERVER_EMBEDDINGS_URL)
        return _model_instance

    logger.warning(
        "LLAMA_SERVER_EMBEDDINGS_URL is not configured. "
        "Semantic retrieval will be disabled; heuristic fallback active."
    )
    return None
