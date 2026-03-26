"""Document comparison engine.

Combines two strategies:
1. **Exact diff** — ``deepdiff`` identifies added/removed/changed text at the
   string level (fast, deterministic).
2. **Semantic diff** — an LLM explains *what* changed and *why it matters*
   (slow, requires API call).
"""

from __future__ import annotations

import logging

from deepdiff import DeepDiff
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Character cap per side when sending texts to the LLM (~2 000 tokens each)
_SEMANTIC_DIFF_TEXT_CAP = 8_000


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class ExactDiffStats(BaseModel):
    """Summary statistics from the deepdiff comparison."""

    values_changed: int = Field(default=0, description="Number of changed values.")
    dictionary_item_added: int = Field(default=0, description="Items added.")
    dictionary_item_removed: int = Field(default=0, description="Items removed.")
    iterable_item_added: int = Field(default=0, description="Iterable items added.")
    iterable_item_removed: int = Field(default=0, description="Iterable items removed.")
    total_changes: int = Field(default=0, description="Total number of detected changes.")


class ComparisonResult(BaseModel):
    """Full comparison result between two documents."""

    doc_a_chars: int = Field(description="Character count of document A.")
    doc_b_chars: int = Field(description="Character count of document B.")
    exact_diff: ExactDiffStats = Field(description="Token-level diff statistics.")
    semantic_summary: str = Field(
        description="LLM-generated narrative of what changed and what it means."
    )
    similarity_assessment: str = Field(
        description="High-level similarity assessment: 'identical', 'minor changes', "
        "'moderate changes', or 'significant changes'.",
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SEMANTIC_DIFF_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a document analyst. You will be given two versions of a document "
            "(Document A and Document B). Your job is to:\n"
            "1. Identify the key differences between them.\n"
            "2. Explain the significance of those differences.\n"
            "3. Assess overall similarity: 'identical', 'minor changes', "
            "'moderate changes', or 'significant changes'.\n\n"
            "Be concise and professional.",
        ),
        (
            "human",
            "### Document A\n{text_a}\n\n### Document B\n{text_b}\n\n"
            "Provide a structured comparison covering: "
            "(a) what was added, (b) what was removed, "
            "(c) what was modified, and (d) overall significance.",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_documents(
    doc_a: Document | list[Document],
    doc_b: Document | list[Document],
) -> ComparisonResult:
    """Compare two documents (or document lists) using exact + semantic diff.

    Args:
        doc_a: First document or list of Documents (e.g., the original version).
        doc_b: Second document or list of Documents (e.g., the revised version).

    Returns:
        A :class:`ComparisonResult` with exact diff statistics and an LLM
        semantic summary.
    """
    text_a = _to_text(doc_a)
    text_b = _to_text(doc_b)

    logger.info(
        "Comparing documents: A=%d chars, B=%d chars.", len(text_a), len(text_b)
    )

    exact = _exact_diff(text_a, text_b)
    semantic, similarity = _semantic_diff(text_a, text_b)

    return ComparisonResult(
        doc_a_chars=len(text_a),
        doc_b_chars=len(text_b),
        exact_diff=exact,
        semantic_summary=semantic,
        similarity_assessment=similarity,
    )


def compare_texts(text_a: str, text_b: str) -> ComparisonResult:
    """Compare two raw text strings directly.

    Convenience wrapper around :func:`compare_documents` for callers that
    already have plain strings.
    """
    return compare_documents(
        Document(page_content=text_a),
        Document(page_content=text_b),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_text(source: Document | list[Document]) -> str:
    if isinstance(source, Document):
        return source.page_content
    return "\n\n".join(d.page_content for d in source if d.page_content.strip())


def _exact_diff(text_a: str, text_b: str) -> ExactDiffStats:
    """Run deepdiff on the two texts split into lines for granular comparison."""
    lines_a = text_a.splitlines()
    lines_b = text_b.splitlines()

    try:
        diff = DeepDiff(lines_a, lines_b, ignore_whitespace_changes=True, verbose_level=0)
    except Exception as exc:  # noqa: BLE001
        logger.warning("deepdiff failed: %s", exc)
        diff = {}

    stats = ExactDiffStats(
        values_changed=len(diff.get("values_changed", {})),
        dictionary_item_added=len(diff.get("dictionary_item_added", {})),
        dictionary_item_removed=len(diff.get("dictionary_item_removed", {})),
        iterable_item_added=len(diff.get("iterable_item_added", {})),
        iterable_item_removed=len(diff.get("iterable_item_removed", {})),
    )
    stats.total_changes = (
        stats.values_changed
        + stats.dictionary_item_added
        + stats.dictionary_item_removed
        + stats.iterable_item_added
        + stats.iterable_item_removed
    )
    return stats


def _semantic_diff(text_a: str, text_b: str) -> tuple[str, str]:
    """Call the LLM to produce a narrative diff and similarity rating.

    Returns:
        Tuple of (semantic_summary, similarity_assessment).
    """
    text_a_capped = text_a[:_SEMANTIC_DIFF_TEXT_CAP]
    text_b_capped = text_b[:_SEMANTIC_DIFF_TEXT_CAP]

    llm = get_llm(temperature=0.0)
    chain = _SEMANTIC_DIFF_PROMPT | llm | StrOutputParser()

    try:
        raw: str = chain.invoke({"text_a": text_a_capped, "text_b": text_b_capped})
    except Exception as exc:  # noqa: BLE001
        logger.error("Semantic diff LLM call failed: %s", exc)
        raw = f"Semantic diff unavailable: {exc}"

    # Extract similarity rating from the narrative
    similarity = _extract_similarity(raw)
    logger.info("Semantic diff complete. Similarity: '%s'.", similarity)
    return raw, similarity


def _extract_similarity(text: str) -> str:
    """Heuristically extract a similarity label from LLM output."""
    lower = text.lower()
    if "identical" in lower:
        return "identical"
    if "significant" in lower:
        return "significant changes"
    if "moderate" in lower:
        return "moderate changes"
    if "minor" in lower:
        return "minor changes"
    return "unknown"
