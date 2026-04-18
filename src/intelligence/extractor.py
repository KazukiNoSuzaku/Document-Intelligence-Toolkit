"""Structured data extraction from documents using LLM + Pydantic schemas.

Uses LangChain's ``with_structured_output`` (tool-calling under the hood) so
the LLM always returns a validated Pydantic model — no manual JSON parsing.

Includes a retry-with-feedback guardrail: if the initial extraction has quality
issues (empty required fields, suspiciously short summary, etc.) the validation
errors are fed back to the LLM for a corrective second attempt.

When no API key is configured, falls back to deterministic rule-based extraction
using regex and keyword heuristics — no LLM required.
"""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator

from src.utils.llm_factory import get_llm, has_api_key

logger = logging.getLogger(__name__)

# Rough character cap: ~12 000 tokens × ~4 chars/token
_EXTRACTION_TEXT_CAP = 48_000

# Maximum retry attempts when quality validation fails
_MAX_RETRIES = 2


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

    @model_validator(mode="after")
    def _strip_whitespace(self) -> DocumentExtraction:
        """Normalise whitespace on string fields."""
        self.title = self.title.strip()
        self.document_type = self.document_type.strip()
        self.summary = self.summary.strip()
        self.parties = [p.strip() for p in self.parties if p.strip()]
        self.dates = [d.strip() for d in self.dates if d.strip()]
        self.key_topics = [t.strip() for t in self.key_topics if t.strip()]
        self.key_clauses = [c.strip() for c in self.key_clauses if c.strip()]
        return self


# ---------------------------------------------------------------------------
# Quality validation (guardrails)
# ---------------------------------------------------------------------------


def validate_extraction(result: DocumentExtraction) -> list[str]:
    """Check extraction quality beyond schema conformance.

    Returns a list of human-readable issue descriptions. An empty list
    means the extraction passed all quality checks.
    """
    issues: list[str] = []

    if not result.title:
        issues.append("title is empty — infer a title from the document content.")
    if not result.document_type:
        issues.append("document_type is empty — classify the document (e.g. contract, report).")
    if not result.summary or len(result.summary) < 30:
        issues.append(
            "summary is too short — provide at least one full sentence describing "
            "the document's purpose and content."
        )
    if not result.key_topics:
        issues.append("key_topics is empty — identify at least one topic or theme.")

    return issues


# ---------------------------------------------------------------------------
# Prompts
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

_RETRY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert document analyst. A previous extraction attempt had quality "
            "issues. Fix ONLY the problems listed below while keeping the rest of the "
            "extraction intact. Be precise and factual — do not invent information.",
        ),
        ("human", "Document text:\n\n{text}"),
        (
            "human",
            "Your previous extraction had these issues:\n{feedback}\n\n"
            "Please produce a corrected extraction that addresses every issue listed above.",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_structured_data(
    documents: list[Document],
    max_retries: int = _MAX_RETRIES,
) -> DocumentExtraction:
    """Extract structured metadata and key information from a list of Documents.

    Includes a guardrail loop: after each LLM call the result is validated for
    quality. If issues are found, the validation feedback is sent back to the
    LLM for a corrective attempt (up to ``max_retries`` times).

    Args:
        documents: LangChain Documents to analyse.
        max_retries: Maximum number of retry attempts on quality failure.

    Returns:
        A :class:`DocumentExtraction` Pydantic model with extracted fields.

    Raises:
        ValueError: If ``documents`` is empty.
        RuntimeError: If the LLM fails to return a valid structured response
            after all attempts.
    """
    if not documents:
        raise ValueError("documents list must not be empty.")

    combined = _prepare_text(documents)

    if not has_api_key():
        logger.info("No API key found — using rule-based extraction fallback.")
        return _rule_based_extract(combined)

    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(DocumentExtraction)

    # --- First attempt ---
    result = _invoke_chain(
        _EXTRACTION_PROMPT | structured_llm, {"text": combined}, attempt=1
    )
    issues = validate_extraction(result)

    if not issues:
        logger.info(
            "Extraction complete (attempt 1): type='%s', parties=%d, topics=%d.",
            result.document_type,
            len(result.parties),
            len(result.key_topics),
        )
        return result

    # --- Retry loop with feedback ---
    for attempt in range(2, max_retries + 2):
        feedback = "\n".join(f"- {issue}" for issue in issues)
        logger.warning(
            "Extraction attempt %d had %d quality issue(s): %s. Retrying with feedback.",
            attempt - 1,
            len(issues),
            feedback,
        )

        retry_chain = _RETRY_PROMPT | structured_llm
        result = _invoke_chain(
            retry_chain,
            {"text": combined, "feedback": feedback},
            attempt=attempt,
        )
        issues = validate_extraction(result)

        if not issues:
            logger.info(
                "Extraction complete (attempt %d): type='%s', parties=%d, topics=%d.",
                attempt,
                result.document_type,
                len(result.parties),
                len(result.key_topics),
            )
            return result

    # Exhausted retries — return best effort with a warning
    remaining = "\n".join(f"- {issue}" for issue in issues)
    logger.warning(
        "Extraction still has issues after %d attempt(s). Returning best effort. "
        "Remaining issues:\n%s",
        max_retries + 1,
        remaining,
    )
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prepare_text(documents: list[Document]) -> str:
    """Concatenate document text with a character cap."""
    combined = "\n\n".join(d.page_content for d in documents if d.page_content.strip())
    if len(combined) > _EXTRACTION_TEXT_CAP:
        logger.warning(
            "Document text truncated from %d to %d characters for extraction. "
            "Consider chunking first.",
            len(combined),
            _EXTRACTION_TEXT_CAP,
        )
        combined = combined[:_EXTRACTION_TEXT_CAP]
    return combined


def _rule_based_extract(text: str) -> DocumentExtraction:
    """Deterministic fallback extraction using regex and keyword heuristics."""

    # --- Title: first non-empty line, capped at 120 chars ---
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0][:120] if lines else "Untitled Document"

    # --- Document type: keyword matching ---
    lower = text.lower()
    type_keywords: list[tuple[str, str]] = [
        ("contract", "contract"),
        ("agreement", "contract"),
        ("invoice", "invoice"),
        ("receipt", "invoice"),
        ("report", "report"),
        ("policy", "policy"),
        ("proposal", "proposal"),
        ("letter", "letter"),
        ("memo", "memo"),
        ("nda", "non-disclosure agreement"),
        ("non-disclosure", "non-disclosure agreement"),
        ("research", "research paper"),
        ("abstract", "research paper"),
    ]
    document_type = "document"
    for keyword, label in type_keywords:
        if keyword in lower:
            document_type = label
            break

    # --- Dates: ISO, US long form, and numeric formats ---
    date_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    ]
    dates: list[str] = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text, re.IGNORECASE))
    dates = list(dict.fromkeys(dates))[:10]  # deduplicate, cap at 10

    # --- Parties: "between X and Y", "by and between X and Y" patterns ---
    parties: list[str] = []
    party_patterns = [
        r"between\s+([A-Z][A-Za-z\s,\.]+?)\s+and\s+([A-Z][A-Za-z\s,\.]+?)(?:\s*[\(\,\.]|$)",
        r"(?:by|with)\s+([A-Z][A-Za-z\s]+(?:Inc|LLC|Ltd|Corp|Company|Co|Group|Solutions|Services)\.?)",
        r"([A-Z][A-Za-z\s]+(?:Inc|LLC|Ltd|Corp|Company|Co|Group|Solutions|Services)\.?)\s+\(",
    ]
    for pattern in party_patterns:
        for match in re.finditer(pattern, text):
            parties.extend(g.strip() for g in match.groups() if g and g.strip())
    parties = list(dict.fromkeys(p for p in parties if 2 < len(p) < 80))[:8]

    # --- Key topics: frequent capitalised noun phrases and domain keywords ---
    domain_terms = [
        "payment", "termination", "liability", "indemnification", "confidentiality",
        "intellectual property", "warranty", "arbitration", "jurisdiction",
        "deliverable", "milestone", "scope", "pricing", "governing law",
        "force majeure", "amendment", "assignment", "data privacy", "gdpr",
        "compliance", "audit", "penalty", "interest", "notice period",
    ]
    key_topics = [term for term in domain_terms if term in lower][:8]
    if not key_topics:
        # Fallback: extract capitalised multi-word phrases
        caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text)
        key_topics = list(dict.fromkeys(caps))[:6]

    # --- Summary: first 3 sentences ---
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    summary = " ".join(sentences[:3]) if sentences else text[:300]

    result = DocumentExtraction(
        title=title,
        document_type=document_type,
        parties=parties,
        dates=dates,
        key_topics=key_topics,
        key_clauses=[],
        summary=summary,
    )
    logger.info(
        "Rule-based extraction complete: type='%s', parties=%d, dates=%d, topics=%d.",
        result.document_type,
        len(result.parties),
        len(result.dates),
        len(result.key_topics),
    )
    return result


def _invoke_chain(chain, inputs: dict, attempt: int) -> DocumentExtraction:
    """Invoke a chain and handle failures gracefully."""
    try:
        result = chain.invoke(inputs)
    except Exception as exc:  # noqa: BLE001
        logger.error("Structured extraction failed on attempt %d: %s", attempt, exc)
        raise RuntimeError(f"Extraction failed on attempt {attempt}: {exc}") from exc

    if result is None:
        raise RuntimeError(
            f"LLM returned None on attempt {attempt}. "
            "The model may not support structured output."
        )

    return result
