"""PDF ingestion using pdfplumber (primary) with pypdf as fallback.

Returns LangChain Document objects so the rest of the pipeline is
parser-agnostic.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_pdf(file_path: str | Path) -> list[Document]:
    """Parse a PDF file and return one LangChain Document per page.

    Tries pdfplumber first (better text + table extraction). Falls back to
    pypdf if pdfplumber raises an exception.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        A list of Document objects, each carrying:
          - ``page_content``: extracted text for that page.
          - ``metadata``: dict with ``source``, ``page`` (0-indexed),
            ``total_pages``, and any PDF-level metadata from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    path = _validate_path(file_path, suffix=".pdf")

    try:
        docs = _load_with_pdfplumber(path)
        logger.info("Loaded %d pages from '%s' via pdfplumber.", len(docs), path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pdfplumber failed for '%s' (%s). Falling back to pypdf.", path.name, exc
        )
        docs = _load_with_pypdf(path)
        logger.info("Loaded %d pages from '%s' via pypdf.", len(docs), path.name)

    return docs


def extract_tables_from_pdf(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract all tables from a PDF as a list of structured dicts.

    Each entry contains the page number and the table data as a list of rows
    (each row is a list of cell strings).

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dicts: ``{"page": int, "table_index": int, "data": list[list[str | None]]}``
    """
    path = _validate_path(file_path, suffix=".pdf")
    tables: list[dict[str, Any]] = []

    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_tables = page.extract_tables()
                    for tbl_idx, table in enumerate(page_tables):
                        tables.append(
                            {
                                "page": page_num,
                                "table_index": tbl_idx,
                                "data": table,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Could not extract tables from page %d of '%s': %s",
                        page_num,
                        path.name,
                        exc,
                    )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to open '%s' for table extraction: %s", path.name, exc)

    return tables


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_path(file_path: str | Path, suffix: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() != suffix:
        raise ValueError(f"Expected a '{suffix}' file, got: {path.suffix}")
    return path


def _load_with_pdfplumber(path: Path) -> list[Document]:
    """Primary extraction strategy using pdfplumber."""
    docs: list[Document] = []

    with pdfplumber.open(path) as pdf:
        pdf_meta = _clean_metadata(pdf.metadata or {})
        total_pages = len(pdf.pages)

        for page_num, page in enumerate(pdf.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not extract text from page %d of '%s': %s",
                    page_num,
                    path.name,
                    exc,
                )
                text = ""

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": page_num,
                        "total_pages": total_pages,
                        **pdf_meta,
                    },
                )
            )

    return docs


def _load_with_pypdf(path: Path) -> list[Document]:
    """Fallback extraction strategy using pypdf."""
    docs: list[Document] = []

    with open(path, "rb") as f:
        reader = PdfReader(f)
        pdf_meta = _clean_metadata(dict(reader.metadata or {}))
        total_pages = len(reader.pages)

        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pypdf could not extract text from page %d of '%s': %s",
                    page_num,
                    path.name,
                    exc,
                )
                text = ""

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "page": page_num,
                        "total_pages": total_pages,
                        **pdf_meta,
                    },
                )
            )

    return docs


def _clean_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Stringify metadata values so they're safe to store in LangChain metadata."""
    return {
        str(k).lstrip("/"): str(v) if v is not None else ""
        for k, v in raw.items()
    }
