"""Document Intelligence Toolkit — Streamlit UI.

Two tabs:
  Tab 1 — Analyse: Upload a single document → summarise + extract structured data.
  Tab 2 — Compare: Upload two documents → exact diff + semantic comparison.

Run with:
    streamlit run ui/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.documents import Document

from src.document_parsers.docx_parser import load_docx
from src.document_parsers.pdf_parser import load_pdf
from src.intelligence.comparator import compare_documents
from src.intelligence.extractor import DocumentExtraction, extract_structured_data
from src.intelligence.summarizer import summarize_documents

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Document Intelligence Toolkit",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACCEPTED_TYPES = ["pdf", "docx"]


def load_uploaded_file(uploaded_file: Any) -> list[Document]:
    """Save an uploaded Streamlit file to a temp path, parse it, then clean up."""
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)

        if suffix == ".pdf":
            return load_pdf(tmp_path)
        elif suffix == ".docx":
            return load_docx(tmp_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def doc_preview(docs: list[Document], max_chars: int = 2_000) -> str:
    """Return a truncated text preview of the documents."""
    text = "\n\n".join(d.page_content for d in docs)
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n… (truncated)"
    return text


def render_extraction(result: DocumentExtraction) -> None:
    """Render a DocumentExtraction as formatted Streamlit widgets."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Title:** {result.title}")
        st.markdown(f"**Type:** {result.document_type}")
        if result.parties:
            st.markdown("**Parties:**")
            for p in result.parties:
                st.markdown(f"- {p}")
        if result.dates:
            st.markdown("**Dates:**")
            for d in result.dates:
                st.markdown(f"- {d}")

    with col2:
        if result.key_topics:
            st.markdown("**Key Topics:**")
            for t in result.key_topics:
                st.markdown(f"- {t}")
        if result.key_clauses:
            st.markdown("**Key Clauses / Obligations:**")
            for c in result.key_clauses:
                st.markdown(f"- {c}")

    st.markdown("**Summary:**")
    st.info(result.summary)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Settings")
    summary_style = st.selectbox(
        "Summary style",
        options=["concise", "detailed", "bullet"],
        index=0,
        help="Controls how the LLM frames the summary.",
    )
    st.markdown("---")
    st.caption(
        "Set `LLM_PROVIDER` (anthropic / openai) and `LLM_MODEL` in your `.env` "
        "to switch models."
    )

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------

st.title("Document Intelligence Toolkit")

tab_analyse, tab_compare = st.tabs(["📋 Analyse", "🔀 Compare"])

# ===========================================================================
# TAB 1 — ANALYSE
# ===========================================================================

with tab_analyse:
    st.subheader("Upload a document to summarise and extract structured data")

    uploaded = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=ACCEPTED_TYPES,
        key="analyse_upload",
    )

    if uploaded is not None:
        with st.spinner("Parsing document…"):
            try:
                docs = load_uploaded_file(uploaded)
            except Exception as exc:
                st.error(f"Failed to parse file: {exc}")
                st.stop()

        total_elements = (
            docs[-1].metadata.get("total_pages")
            or docs[-1].metadata.get("total_sections")
            or len(docs)
        )
        st.success(
            f"Loaded **{uploaded.name}** — "
            f"{len(docs)} section(s) / page(s), "
            f"{sum(len(d.page_content) for d in docs):,} chars"
        )

        with st.expander("Document preview", expanded=False):
            st.text(doc_preview(docs))

        st.markdown("---")

        col_sum, col_ext = st.columns(2, gap="large")

        # ---- Summarisation ----
        with col_sum:
            st.markdown("### Summary")
            if st.button("Generate Summary", key="btn_summary"):
                with st.spinner("Summarising…"):
                    try:
                        summary = summarize_documents(docs, style=summary_style)
                        st.session_state["summary"] = summary
                    except Exception as exc:
                        st.error(f"Summarisation failed: {exc}")

            if "summary" in st.session_state:
                st.write(st.session_state["summary"])

        # ---- Extraction ----
        with col_ext:
            st.markdown("### Structured Extraction")
            if st.button("Extract Data", key="btn_extract"):
                with st.spinner("Extracting…"):
                    try:
                        extraction = extract_structured_data(docs)
                        st.session_state["extraction"] = extraction
                    except Exception as exc:
                        st.error(f"Extraction failed: {exc}")

            if "extraction" in st.session_state:
                render_extraction(st.session_state["extraction"])

# ===========================================================================
# TAB 2 — COMPARE
# ===========================================================================

with tab_compare:
    st.subheader("Upload two documents to compare them")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("#### Document A (original)")
        file_a = st.file_uploader(
            "Choose PDF or DOCX",
            type=ACCEPTED_TYPES,
            key="compare_upload_a",
        )

    with col_b:
        st.markdown("#### Document B (revised)")
        file_b = st.file_uploader(
            "Choose PDF or DOCX",
            type=ACCEPTED_TYPES,
            key="compare_upload_b",
        )

    if file_a is not None and file_b is not None:
        with st.spinner("Parsing documents…"):
            try:
                docs_a = load_uploaded_file(file_a)
                docs_b = load_uploaded_file(file_b)
            except Exception as exc:
                st.error(f"Failed to parse file: {exc}")
                st.stop()

        st.success(
            f"**A:** {file_a.name} ({len(docs_a)} section(s))   |   "
            f"**B:** {file_b.name} ({len(docs_b)} section(s))"
        )

        with st.expander("Preview Document A", expanded=False):
            st.text(doc_preview(docs_a))
        with st.expander("Preview Document B", expanded=False):
            st.text(doc_preview(docs_b))

        st.markdown("---")

        if st.button("Run Comparison", key="btn_compare"):
            with st.spinner("Comparing documents…"):
                try:
                    result = compare_documents(docs_a, docs_b)
                    st.session_state["compare_result"] = result
                except Exception as exc:
                    st.error(f"Comparison failed: {exc}")

        if "compare_result" in st.session_state:
            res = st.session_state["compare_result"]

            # ---- Exact diff stats ----
            st.markdown("### Exact Diff Statistics")
            stats = res.exact_diff
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Changes", stats.total_changes)
            m2.metric("Lines Added", stats.iterable_item_added)
            m3.metric("Lines Removed", stats.iterable_item_removed)
            m4.metric("Lines Modified", stats.values_changed)

            # ---- Similarity badge ----
            similarity_colours = {
                "identical": "green",
                "minor changes": "blue",
                "moderate changes": "orange",
                "significant changes": "red",
                "unknown": "grey",
            }
            colour = similarity_colours.get(res.similarity_assessment, "grey")
            st.markdown(
                f"**Overall Similarity:** "
                f":{colour}[{res.similarity_assessment.upper()}]"
            )

            # ---- Semantic summary ----
            st.markdown("### Semantic Analysis")
            st.write(res.semantic_summary)

    elif file_a is not None or file_b is not None:
        st.info("Please upload both Document A and Document B to run a comparison.")
