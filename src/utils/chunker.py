"""Document chunking utilities.

Wraps LangChain's RecursiveCharacterTextSplitter with token-aware sizing so
that chunks respect both document boundaries (paragraph/section breaks) and
LLM context-window limits.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# Separator hierarchy: try splitting on section breaks, then paragraphs, then
# sentences, then words — never mid-word.
_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]


def _default_chunk_size() -> int:
    return int(os.getenv("CHUNK_SIZE", "1500"))


def _default_chunk_overlap() -> int:
    return int(os.getenv("CHUNK_OVERLAP", "150"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split a list of Documents into token-bounded chunks.

    Each input Document is split independently so metadata (source, page,
    section) is correctly propagated to every chunk. The ``chunk_index`` and
    ``total_chunks`` fields are added to metadata for downstream traceability.

    Args:
        documents: LangChain Documents to split.
        chunk_size: Max tokens per chunk. Reads ``CHUNK_SIZE`` env var if not
            provided (default 1500).
        chunk_overlap: Token overlap between consecutive chunks. Reads
            ``CHUNK_OVERLAP`` env var if not provided (default 150).

    Returns:
        Flat list of chunked Documents with enriched metadata.
    """
    size = chunk_size if chunk_size is not None else _default_chunk_size()
    overlap = chunk_overlap if chunk_overlap is not None else _default_chunk_overlap()

    splitter = _build_splitter(size, overlap)
    all_chunks: list[Document] = []

    for doc in documents:
        try:
            chunks = splitter.split_documents([doc])
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to chunk document (source=%s): %s. Using document as-is.",
                doc.metadata.get("source", "unknown"),
                exc,
            )
            chunks = [doc]

        total = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_total"] = total
            all_chunks.append(chunk)

    logger.info(
        "Chunked %d document(s) → %d chunks (size=%d, overlap=%d).",
        len(documents),
        len(all_chunks),
        size,
        overlap,
    )
    return all_chunks


def chunk_text(
    text: str,
    metadata: dict[str, Any] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split a raw string into token-bounded Document chunks.

    Convenience wrapper for callers that have plain text rather than a
    pre-built Document.

    Args:
        text: Input text to split.
        metadata: Optional metadata dict attached to every resulting chunk.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Token overlap between consecutive chunks.

    Returns:
        List of Document objects.
    """
    doc = Document(page_content=text, metadata=metadata or {})
    return chunk_documents([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def estimate_chunk_count(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    """Estimate how many chunks a given text will produce without splitting it.

    Useful for deciding whether to use a Map-Reduce summarization strategy.

    Args:
        text: The text to estimate chunk count for.
        chunk_size: Max tokens per chunk.
        chunk_overlap: Token overlap.

    Returns:
        Estimated number of chunks (always >= 1).
    """
    size = chunk_size if chunk_size is not None else _default_chunk_size()
    overlap = chunk_overlap if chunk_overlap is not None else _default_chunk_overlap()
    total_tokens = count_tokens(text)

    if total_tokens <= size:
        return 1

    stride = size - overlap
    if stride <= 0:
        return 1

    return max(1, -(-total_tokens // stride))  # ceiling division


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Build a token-aware RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
    )
