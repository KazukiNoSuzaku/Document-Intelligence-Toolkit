"""Structured data extraction from documents using LLM + Pydantic schemas.

Uses LangChain's ``with_structured_output`` (tool-calling under the hood) so
the LLM always returns a validated Pydantic model — no manual JSON parsing.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.utils.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Rough character cap: ~12 000 tokens × ~4 chars/token
_EXTRACTION_TEXT_CAP = 48_000


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class DocumentExtraction(BaseModel):
    """Structured information extracted from a document."""

    title: str = Field(description="Document title or best inferred title.")
    document_type: str = Field(
        description=(
            "Type of document, e.g. contract, report, invoice, policy, "
            "research paper, letter, or other."
        )
    )
    parties: list[str] = Field(
        default_factory=list,
        description="People or organisations mentioned as parties, authors, or signatories.",
    )
    dates: list[str] = Field(
        default_factory=list,
        description="Important dates mentioned in the document (ISO-8601 preferred).",
    )
    key_topics: list[str] = Field(
        default_factory=list,
        description="Main topics, themes, or subject areas covered.",
    )
    key_clauses: list[str] = Field(
        default_factory=list,
        description=(
            "Critical obligations, terms, or clauses (most relevant for contracts/policies)."
        ),
    )
    summary: str = Field(description="One-paragraph summary of the document's purpose and content.")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert document analyst. Extract structured information from the "
            "document text provided by the user. Be precise and factual — do not invent "
            "information that is not present in the text.",
        ),
        ("human", "Extract structured information from the following document:\n\n{text}"),
    ]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_structured_data(documents: list[Document]) -> DocumentExtraction:
    """Extract structured metadata and key information from a list of Documents.

    All documents are concatenated (up to ~12 000 tokens of combined text) and
    passed to the LLM in a single extraction call.  For very large corpora,
    chunk first with :func:`src.utils.chunker.chunk_documents` and pass a
    representative subset.

    Args:
        documents: LangChain Documents to analyse.

    Returns:
        A :class:`DocumentExtraction` Pydantic model with extracted fields.

    Raises:
        ValueError: If ``documents`` is empty.
        RuntimeError: If the LLM fails to return a valid structured response.
    """
    if not documents:
        raise ValueError("documents list must not be empty.")

    # Concatenate text; cap at ~12 000 tokens worth of characters to stay
    # within a single prompt without requiring map-reduce here.
    combined = "\n\n".join(d.page_content for d in documents if d.page_content.strip())
    if len(combined) > _EXTRACTION_TEXT_CAP:
        logger.warning(
            "Document text truncated from %d to %d characters for extraction. "
            "Consider chunking first.",
            len(combined),
            _EXTRACTION_TEXT_CAP,
        )
        combined = combined[:_EXTRACTION_TEXT_CAP]

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(DocumentExtraction)
    chain = _EXTRACTION_PROMPT | structured_llm

    try:
        result: DocumentExtraction = chain.invoke({"text": combined})
    except Exception as exc:  # noqa: BLE001
        logger.error("Structured extraction failed: %s", exc)
        raise RuntimeError(f"Extraction failed: {exc}") from exc

    logger.info(
        "Extraction complete: type='%s', parties=%d, topics=%d.",
        result.document_type,
        len(result.parties),
        len(result.key_topics),
    )
    return result
