"""Token counting utilities backed by tiktoken.

Provides a simple, reusable function for counting tokens so the rest of the
pipeline can make informed decisions about chunking and prompt assembly without
hitting API token limits.
"""

from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)

# Default encoding — cl100k_base covers GPT-4, GPT-3.5, and Anthropic models
# via the same approximate count.
_DEFAULT_ENCODING = "cl100k_base"
_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def get_encoding(encoding_name: str = _DEFAULT_ENCODING) -> tiktoken.Encoding:
    """Return a cached tiktoken encoding by name."""
    if encoding_name not in _ENCODING_CACHE:
        try:
            _ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load encoding '%s' (%s). Falling back to cl100k_base.",
                encoding_name,
                exc,
            )
            _ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(_DEFAULT_ENCODING)
    return _ENCODING_CACHE[encoding_name]


def count_tokens(text: str, encoding_name: str = _DEFAULT_ENCODING) -> int:
    """Count the number of tokens in *text*.

    Args:
        text: Input string.
        encoding_name: tiktoken encoding name (default: ``cl100k_base``).

    Returns:
        Integer token count.
    """
    enc = get_encoding(encoding_name)
    return len(enc.encode(text))


def count_tokens_for_documents(
    documents: list,  # list[langchain_core.documents.Document]
    encoding_name: str = _DEFAULT_ENCODING,
) -> int:
    """Sum token counts across a list of LangChain Document objects.

    Args:
        documents: List of LangChain ``Document`` instances.
        encoding_name: tiktoken encoding name.

    Returns:
        Total token count across all documents.
    """
    return sum(count_tokens(doc.page_content, encoding_name) for doc in documents)
