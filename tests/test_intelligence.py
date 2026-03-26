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
from src.intelligence.extractor import (
    DocumentExtraction,
    extract_structured_data,
    validate_extraction,
)
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
# extractor — validate_extraction (guardrails)
# ---------------------------------------------------------------------------


class TestValidateExtraction:
    def _good_extraction(self) -> DocumentExtraction:
        return DocumentExtraction(
            title="Service Agreement",
            document_type="contract",
            parties=["Acme Corp"],
            dates=["2024-01-15"],
            key_topics=["software development"],
            key_clauses=["payment terms"],
            summary="A software development contract between two companies for Q1 delivery.",
        )

    def test_valid_extraction_returns_no_issues(self) -> None:
        assert validate_extraction(self._good_extraction()) == []

    def test_empty_title_flagged(self) -> None:
        obj = self._good_extraction()
        obj.title = ""
        issues = validate_extraction(obj)
        assert any("title" in i for i in issues)

    def test_empty_document_type_flagged(self) -> None:
        obj = self._good_extraction()
        obj.document_type = ""
        issues = validate_extraction(obj)
        assert any("document_type" in i for i in issues)

    def test_short_summary_flagged(self) -> None:
        obj = self._good_extraction()
        obj.summary = "Short."
        issues = validate_extraction(obj)
        assert any("summary" in i for i in issues)

    def test_empty_key_topics_flagged(self) -> None:
        obj = self._good_extraction()
        obj.key_topics = []
        issues = validate_extraction(obj)
        assert any("key_topics" in i for i in issues)

    def test_multiple_issues_all_reported(self) -> None:
        obj = DocumentExtraction(
            title="",
            document_type="",
            summary="x",
            key_topics=[],
        )
        issues = validate_extraction(obj)
        assert len(issues) == 4


class TestWhitespaceStripping:
    def test_model_validator_strips_fields(self) -> None:
        obj = DocumentExtraction(
            title="  My Title  ",
            document_type="  report  ",
            parties=["  Alice  ", "  ", "Bob"],
            dates=["  2024-01-01  "],
            key_topics=["  AI  "],
            key_clauses=["  clause 1  "],
            summary="  A summary.  ",
        )
        assert obj.title == "My Title"
        assert obj.document_type == "report"
        assert obj.parties == ["Alice", "Bob"]
        assert obj.dates == ["2024-01-01"]
        assert obj.key_topics == ["AI"]
        assert obj.key_clauses == ["clause 1"]
        assert obj.summary == "A summary."

    def test_empty_strings_removed_from_lists(self) -> None:
        obj = DocumentExtraction(
            title="Title",
            document_type="report",
            parties=["", "  ", "Alice"],
            key_topics=["topic"],
            summary="A valid summary sentence for the document.",
        )
        assert obj.parties == ["Alice"]


class TestRetryWithFeedback:
    @patch("src.intelligence.extractor.get_llm")
    def test_retries_on_quality_failure_then_succeeds(self, mock_get_llm: MagicMock) -> None:
        """LLM returns a bad extraction first, then a good one on retry."""
        bad_result = DocumentExtraction(
            title="",
            document_type="contract",
            parties=["Acme Corp"],
            dates=[],
            key_topics=["software"],
            key_clauses=[],
            summary="A contract.",  # too short
        )
        good_result = DocumentExtraction(
            title="Software Services Agreement",
            document_type="contract",
            parties=["Acme Corp"],
            dates=["2024-01-15"],
            key_topics=["software development"],
            key_clauses=["payment terms"],
            summary="A software services contract between Acme Corp and a provider.",
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()

        # First call returns bad, second returns good
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = [bad_result, good_result]
        mock_structured.__or__ = MagicMock(return_value=mock_chain)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        with patch("src.intelligence.extractor._EXTRACTION_PROMPT") as mp1, \
             patch("src.intelligence.extractor._RETRY_PROMPT") as mp2:
            mp1.__or__ = MagicMock(return_value=mock_chain)
            mp2.__or__ = MagicMock(return_value=mock_chain)
            result = extract_structured_data([SHORT_DOC], max_retries=2)

        assert result.title == "Software Services Agreement"
        assert mock_chain.invoke.call_count == 2

    @patch("src.intelligence.extractor.get_llm")
    def test_returns_best_effort_after_max_retries(self, mock_get_llm: MagicMock) -> None:
        """If all retries still fail quality checks, return the last result."""
        bad_result = DocumentExtraction(
            title="",
            document_type="",
            parties=[],
            dates=[],
            key_topics=[],
            key_clauses=[],
            summary="Bad.",
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = bad_result
        mock_structured.__or__ = MagicMock(return_value=mock_chain)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        with patch("src.intelligence.extractor._EXTRACTION_PROMPT") as mp1, \
             patch("src.intelligence.extractor._RETRY_PROMPT") as mp2:
            mp1.__or__ = MagicMock(return_value=mock_chain)
            mp2.__or__ = MagicMock(return_value=mock_chain)
            result = extract_structured_data([SHORT_DOC], max_retries=1)

        # Should have tried initial + 1 retry = 2 calls
        assert mock_chain.invoke.call_count == 2
        # Still returns the best-effort result
        assert isinstance(result, DocumentExtraction)

    @patch("src.intelligence.extractor.get_llm")
    def test_no_retry_when_first_attempt_passes(self, mock_get_llm: MagicMock) -> None:
        """If the first attempt passes validation, no retry should occur."""
        good_result = DocumentExtraction(
            title="Agreement",
            document_type="contract",
            parties=["Acme"],
            dates=["2024-01-15"],
            key_topics=["software"],
            key_clauses=[],
            summary="A full and detailed summary of the agreement between parties.",
        )

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = good_result
        mock_structured.__or__ = MagicMock(return_value=mock_chain)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        with patch("src.intelligence.extractor._EXTRACTION_PROMPT") as mp:
            mp.__or__ = MagicMock(return_value=mock_chain)
            result = extract_structured_data([SHORT_DOC])

        assert mock_chain.invoke.call_count == 1
        assert result.title == "Agreement"

    @patch("src.intelligence.extractor.get_llm")
    def test_raises_if_llm_returns_none(self, mock_get_llm: MagicMock) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = None
        mock_structured.__or__ = MagicMock(return_value=mock_chain)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_get_llm.return_value = mock_llm

        with patch("src.intelligence.extractor._EXTRACTION_PROMPT") as mp:
            mp.__or__ = MagicMock(return_value=mock_chain)
            with pytest.raises(RuntimeError, match="returned None"):
                extract_structured_data([SHORT_DOC])


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
