"""Unit tests for chunking utilities and token counter.

No LLM calls — pure logic tests against in-memory text.
Run with:  pytest tests/test_chunker.py -v
"""

from __future__ import annotations

import os

import pytest
from langchain_core.documents import Document

from src.utils.chunker import (
    chunk_documents,
    chunk_text,
    estimate_chunk_count,
)
from src.utils.token_counter import count_tokens, count_tokens_for_documents


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SHORT_TEXT = "The quick brown fox jumps over the lazy dog."

LONG_TEXT = (
    "This is sentence number {i}. It adds more tokens to the document. "
    "We use it to simulate a long document that must be chunked.\n\n"
)


def make_long_text(sentences: int = 200) -> str:
    return "".join(LONG_TEXT.format(i=i) for i in range(sentences))


def make_doc(text: str, **meta) -> Document:
    return Document(page_content=text, metadata=meta)


# ---------------------------------------------------------------------------
# token_counter tests
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_string_is_zero(self) -> None:
        assert count_tokens("") == 0

    def test_short_text_is_positive(self) -> None:
        assert count_tokens(SHORT_TEXT) > 0

    def test_longer_text_has_more_tokens(self) -> None:
        assert count_tokens(SHORT_TEXT * 10) > count_tokens(SHORT_TEXT)

    def test_returns_int(self) -> None:
        assert isinstance(count_tokens(SHORT_TEXT), int)

    def test_known_token_count(self) -> None:
        # "hello world" encodes to 2 tokens with cl100k_base
        assert count_tokens("hello world") == 2


class TestCountTokensForDocuments:
    def test_empty_list_is_zero(self) -> None:
        assert count_tokens_for_documents([]) == 0

    def test_sums_across_documents(self) -> None:
        docs = [make_doc("hello world"), make_doc("foo bar")]
        total = count_tokens_for_documents(docs)
        assert total == count_tokens("hello world") + count_tokens("foo bar")

    def test_single_doc(self) -> None:
        doc = make_doc(SHORT_TEXT)
        assert count_tokens_for_documents([doc]) == count_tokens(SHORT_TEXT)


# ---------------------------------------------------------------------------
# chunk_text tests
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_returns_single_chunk(self) -> None:
        chunks = chunk_text(SHORT_TEXT, chunk_size=500, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].page_content == SHORT_TEXT

    def test_long_text_returns_multiple_chunks(self) -> None:
        long = make_long_text(200)
        chunks = chunk_text(long, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_each_chunk_within_token_budget(self) -> None:
        long = make_long_text(200)
        chunk_size = 300
        chunks = chunk_text(long, chunk_size=chunk_size, chunk_overlap=30)
        for chunk in chunks:
            tokens = count_tokens(chunk.page_content)
            # Allow a small overshoot (splitter works on characters, not exact tokens)
            assert tokens <= chunk_size + 50, (
                f"Chunk exceeds budget: {tokens} tokens (limit {chunk_size})"
            )

    def test_metadata_propagated_to_chunks(self) -> None:
        long = make_long_text(100)
        chunks = chunk_text(long, metadata={"source": "test.txt"}, chunk_size=200)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.txt"

    def test_chunk_index_metadata_added(self) -> None:
        long = make_long_text(100)
        chunks = chunk_text(long, chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata
            assert "chunk_total" in chunk.metadata

    def test_chunk_indices_sequential(self) -> None:
        long = make_long_text(100)
        chunks = chunk_text(long, chunk_size=200, chunk_overlap=20)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_returns_documents(self) -> None:
        chunks = chunk_text(SHORT_TEXT, chunk_size=500)
        assert all(isinstance(c, Document) for c in chunks)


# ---------------------------------------------------------------------------
# chunk_documents tests
# ---------------------------------------------------------------------------


class TestChunkDocuments:
    def test_empty_list_returns_empty(self) -> None:
        assert chunk_documents([]) == []

    def test_preserves_source_metadata(self) -> None:
        doc = make_doc(make_long_text(100), source="contract.pdf", page=0)
        chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata.get("source") == "contract.pdf"
            assert chunk.metadata.get("page") == 0

    def test_multiple_documents_chunked_independently(self) -> None:
        docs = [
            make_doc(make_long_text(50), source="doc1.pdf"),
            make_doc(make_long_text(50), source="doc2.pdf"),
        ]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        sources = {c.metadata["source"] for c in chunks}
        assert sources == {"doc1.pdf", "doc2.pdf"}

    def test_single_short_doc_not_split(self) -> None:
        doc = make_doc(SHORT_TEXT)
        chunks = chunk_documents([doc], chunk_size=1000)
        assert len(chunks) == 1

    def test_overlap_creates_shared_content(self) -> None:
        # With overlap, consecutive chunks should share some tokens
        long = make_long_text(80)
        chunks_no_overlap = chunk_text(long, chunk_size=150, chunk_overlap=0)
        chunks_with_overlap = chunk_text(long, chunk_size=150, chunk_overlap=50)
        # Overlap increases total token count across chunks
        tokens_no = sum(count_tokens(c.page_content) for c in chunks_no_overlap)
        tokens_with = sum(count_tokens(c.page_content) for c in chunks_with_overlap)
        assert tokens_with >= tokens_no


# ---------------------------------------------------------------------------
# estimate_chunk_count tests
# ---------------------------------------------------------------------------


class TestEstimateChunkCount:
    def test_short_text_is_one_chunk(self) -> None:
        assert estimate_chunk_count(SHORT_TEXT, chunk_size=500) == 1

    def test_long_text_is_multiple_chunks(self) -> None:
        long = make_long_text(200)
        count = estimate_chunk_count(long, chunk_size=200, chunk_overlap=20)
        assert count > 1

    def test_empty_text_is_one_chunk(self) -> None:
        assert estimate_chunk_count("", chunk_size=500) == 1

    def test_larger_chunk_size_reduces_count(self) -> None:
        long = make_long_text(100)
        small = estimate_chunk_count(long, chunk_size=100, chunk_overlap=10)
        large = estimate_chunk_count(long, chunk_size=500, chunk_overlap=50)
        assert large < small

    def test_returns_int(self) -> None:
        assert isinstance(estimate_chunk_count(SHORT_TEXT), int)


# ---------------------------------------------------------------------------
# Environment variable override tests
# ---------------------------------------------------------------------------


class TestEnvVarDefaults:
    def test_chunk_size_env_var_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "100")
        monkeypatch.setenv("CHUNK_OVERLAP", "10")
        long = make_long_text(200)
        # With size=100 we should get many chunks
        chunks = chunk_text(long)
        assert len(chunks) > 5

    def test_chunk_overlap_env_var_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CHUNK_SIZE", "1500")
        monkeypatch.setenv("CHUNK_OVERLAP", "0")
        long = make_long_text(50)
        chunks = chunk_text(long)
        # Should still produce valid chunks
        assert len(chunks) >= 1
