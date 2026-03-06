"""Unit tests for intelligence modules: summarizer, extractor, comparator.

All LLM calls are mocked — no API keys or network access required.
Run with:  pytest tests/test_intelligence.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.intelligence.comparator import (
    ComparisonResult,
    ExactDiffStats,
    _exact_diff,
    _extract_similarity,
    _to_text,
    compare_documents,
    compare_texts,
)
from src.intelligence.extractor import DocumentExtraction, extract_structured_data
from src.intelligence.summarizer import summarize_documents


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_doc(text: str, **meta) -> Document:
    return Document(page_content=text, metadata=meta)


SHORT_DOC = make_doc(
    "This agreement is made between Acme Corp and Globex Inc on 2024-01-15. "
    "The scope covers software development services for Q1 2024.",
    source="contract.pdf",
)

LONG_TEXT = " ".join([f"Sentence {i} about document intelligence." for i in range(500)])
LONG_DOC = make_doc(LONG_TEXT, source="long.pdf")


# ---------------------------------------------------------------------------
# summarizer tests
# ---------------------------------------------------------------------------


class TestSummarizeDocuments:
    def test_empty_list_returns_empty_string(self) -> None:
        assert summarize_documents([]) == ""

    @patch("src.intelligence.summarizer.get_llm")
    def test_single_pass_for_short_doc(self, mock_get_llm: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_llm.__or__ = lambda self, other: MagicMock(
            __or__=lambda s, o: MagicMock(invoke=lambda x: "Short summary.")
        )
        mock_get_llm.return_value = mock_llm

        # Build a proper chain mock that returns a string
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Short summary."

        with patch("src.intelligence.summarizer._DIRECT_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            # Force single-pass by setting a very high threshold
            result = summarize_documents([SHORT_DOC], map_reduce_threshold=999_999)

        assert isinstance(result, str)

    @patch("src.intelligence.summarizer.get_llm")
    def test_returns_string(self, mock_get_llm: MagicMock) -> None:
        """Verify the function always returns a string."""
        # Set up a chain mock
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "A fine summary."

        with patch("src.intelligence.summarizer._DIRECT_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = summarize_documents([SHORT_DOC], map_reduce_threshold=999_999)

        assert isinstance(result, str)

    def test_style_options_accepted(self) -> None:
        """All defined style keys must be accepted without raising."""
        with patch("src.intelligence.summarizer.get_llm") as mock_get_llm:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "ok"
            with patch("src.intelligence.summarizer._DIRECT_PROMPT") as mock_prompt:
                mock_prompt.__or__ = MagicMock(return_value=mock_chain)
                for style in ("concise", "detailed", "bullet"):
                    result = summarize_documents(
                        [SHORT_DOC], style=style, map_reduce_threshold=999_999
                    )
                    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# extractor tests
# ---------------------------------------------------------------------------


class TestExtractStructuredData:
    def test_raises_for_empty_documents(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            extract_structured_data([])

    @patch("src.intelligence.extractor.get_llm")
    def test_returns_document_extraction_model(self, mock_get_llm: MagicMock) -> None:
        expected = DocumentExtraction(
            title="Service Agreement",
            document_type="contract",
            parties=["Acme Corp", "Globex Inc"],
            dates=["2024-01-15"],
            key_topics=["software development", "services"],
            key_clauses=["payment terms", "termination"],
            summary="A software development contract between two companies.",
        )
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = expected
        mock_structured.__or__ = MagicMock(return_value=mock_chain)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        with patch("src.intelligence.extractor._EXTRACTION_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = extract_structured_data([SHORT_DOC])

        assert isinstance(result, DocumentExtraction)

    @patch("src.intelligence.extractor.get_llm")
    def test_raises_runtime_error_on_llm_failure(self, mock_get_llm: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API error")
        mock_structured.__or__ = MagicMock(return_value=mock_chain)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        with patch("src.intelligence.extractor._EXTRACTION_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            with pytest.raises(RuntimeError, match="Extraction failed"):
                extract_structured_data([SHORT_DOC])

    def test_document_extraction_schema_fields(self) -> None:
        """Pydantic model must have all required fields with correct types."""
        obj = DocumentExtraction(
            title="Test",
            document_type="report",
            parties=[],
            dates=[],
            key_topics=[],
            key_clauses=[],
            summary="A test document.",
        )
        assert obj.title == "Test"
        assert obj.document_type == "report"
        assert obj.parties == []
        assert isinstance(obj.summary, str)


# ---------------------------------------------------------------------------
# comparator — _to_text helper
# ---------------------------------------------------------------------------


class TestToText:
    def test_single_document(self) -> None:
        doc = make_doc("Hello world")
        assert _to_text(doc) == "Hello world"

    def test_list_of_documents_joined(self) -> None:
        docs = [make_doc("Part one."), make_doc("Part two.")]
        text = _to_text(docs)
        assert "Part one." in text
        assert "Part two." in text

    def test_empty_content_skipped(self) -> None:
        docs = [make_doc(""), make_doc("Real content."), make_doc("   ")]
        text = _to_text(docs)
        assert "Real content." in text
        # Empty / whitespace-only docs should not contribute double newlines
        assert text.strip() == "Real content."


# ---------------------------------------------------------------------------
# comparator — _exact_diff helper
# ---------------------------------------------------------------------------


class TestExactDiff:
    def test_identical_texts_have_zero_changes(self) -> None:
        stats = _exact_diff("hello world", "hello world")
        assert stats.total_changes == 0

    def test_different_texts_have_nonzero_changes(self) -> None:
        stats = _exact_diff("line one\nline two", "line one\nline three")
        assert stats.total_changes > 0

    def test_returns_exact_diff_stats(self) -> None:
        stats = _exact_diff("a", "b")
        assert isinstance(stats, ExactDiffStats)

    def test_added_lines_detected(self) -> None:
        stats = _exact_diff("line one", "line one\nline two")
        assert stats.iterable_item_added > 0

    def test_removed_lines_detected(self) -> None:
        stats = _exact_diff("line one\nline two", "line one")
        assert stats.iterable_item_removed > 0

    def test_total_is_sum_of_parts(self) -> None:
        stats = _exact_diff("x\ny", "x\nz\nw")
        expected = (
            stats.values_changed
            + stats.dictionary_item_added
            + stats.dictionary_item_removed
            + stats.iterable_item_added
            + stats.iterable_item_removed
        )
        assert stats.total_changes == expected


# ---------------------------------------------------------------------------
# comparator — _extract_similarity helper
# ---------------------------------------------------------------------------


class TestExtractSimilarity:
    def test_identical(self) -> None:
        assert _extract_similarity("The documents are identical.") == "identical"

    def test_significant(self) -> None:
        assert _extract_similarity("There are significant changes between the two.") == "significant changes"

    def test_moderate(self) -> None:
        assert _extract_similarity("Moderate changes were found.") == "moderate changes"

    def test_minor(self) -> None:
        assert _extract_similarity("Only minor changes detected.") == "minor changes"

    def test_unknown_falls_back(self) -> None:
        assert _extract_similarity("Some differences exist.") == "unknown"


# ---------------------------------------------------------------------------
# comparator — compare_documents / compare_texts (mocked LLM)
# ---------------------------------------------------------------------------


class TestCompareDocuments:
    @patch("src.intelligence.comparator.get_llm")
    def test_returns_comparison_result(self, mock_get_llm: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "There are minor changes between the two versions."
        with patch("src.intelligence.comparator._SEMANTIC_DIFF_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = compare_texts("Hello world.", "Hello world!")

        assert isinstance(result, ComparisonResult)
        assert result.doc_a_chars == len("Hello world.")
        assert result.doc_b_chars == len("Hello world!")

    @patch("src.intelligence.comparator.get_llm")
    def test_identical_texts_zero_exact_changes(self, mock_get_llm: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The documents are identical."
        with patch("src.intelligence.comparator._SEMANTIC_DIFF_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = compare_texts("same text", "same text")

        assert result.exact_diff.total_changes == 0

    @patch("src.intelligence.comparator.get_llm")
    def test_compare_documents_accepts_lists(self, mock_get_llm: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "minor changes"
        with patch("src.intelligence.comparator._SEMANTIC_DIFF_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            docs_a = [make_doc("Section 1."), make_doc("Section 2.")]
            docs_b = [make_doc("Section 1."), make_doc("Section 2 revised.")]
            result = compare_documents(docs_a, docs_b)

        assert isinstance(result, ComparisonResult)
        assert result.exact_diff.total_changes > 0

    @patch("src.intelligence.comparator.get_llm")
    def test_similarity_assessment_populated(self, mock_get_llm: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "There are significant changes in scope and pricing."
        with patch("src.intelligence.comparator._SEMANTIC_DIFF_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = compare_texts("Old contract text.", "Completely new contract text.")

        assert result.similarity_assessment != ""

    @patch("src.intelligence.comparator.get_llm")
    def test_semantic_summary_is_string(self, mock_get_llm: MagicMock) -> None:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "moderate changes detected"
        with patch("src.intelligence.comparator._SEMANTIC_DIFF_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            result = compare_texts("doc v1", "doc v2")

        assert isinstance(result.semantic_summary, str)
        assert len(result.semantic_summary) > 0
