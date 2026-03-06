"""Document summarization using Map-Reduce for long documents.

Short documents (below ``MAP_REDUCE_THRESHOLD`` tokens) are summarised in a
single LLM pass.  Longer documents are split into chunks, each chunk is
summarised independently (map step), then all chunk summaries are collapsed
into a final summary (reduce step).
"""

from __future__ import annotations

import logging
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.utils.chunker import chunk_documents
from src.utils.llm_factory import get_llm
from src.utils.token_counter import count_tokens_for_documents

logger = logging.getLogger(__name__)

# Documents with more tokens than this threshold use Map-Reduce instead of a
# single-pass summary.  Override via the MAP_REDUCE_THRESHOLD env var.
_DEFAULT_THRESHOLD = 6_000

_MAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Summarise the following document section {style}. "
            "Be factual and preserve important details.\n\n{text}",
        )
    ]
)

_REDUCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "The following are summaries of individual sections from a larger document. "
            "Combine them into a single coherent summary {style}. "
            "Do not repeat yourself and keep the summary well-structured.\n\n{text}",
        )
    ]
)

_DIRECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Summarise the following document {style}. "
            "Be factual and preserve important details.\n\n{text}",
        )
    ]
)

_STYLE_PHRASES: dict[str, str] = {
    "concise": "in 3-5 sentences",
    "detailed": "in detail, covering all major points",
    "bullet": "as a bullet-point list of key points",
}


def summarize_documents(
    documents: list[Document],
    style: str = "concise",
    map_reduce_threshold: int | None = None,
) -> str:
    """Summarise a list of Documents into a single string.

    Automatically chooses between a single-pass and a Map-Reduce strategy
    based on total document token count.

    Args:
        documents: LangChain Documents to summarise (e.g., from a parser).
        style: Summary style — ``"concise"`` (default), ``"detailed"``, or
            ``"bullet"``.
        map_reduce_threshold: Token count above which Map-Reduce is used.
            Reads ``MAP_REDUCE_THRESHOLD`` env var if not provided
            (default 6 000).

    Returns:
        Summary string produced by the LLM.
    """
    if not documents:
        return ""

    threshold = (
        map_reduce_threshold
        if map_reduce_threshold is not None
        else int(os.getenv("MAP_REDUCE_THRESHOLD", str(_DEFAULT_THRESHOLD)))
    )
    style_phrase = _STYLE_PHRASES.get(style, style)
    total_tokens = count_tokens_for_documents(documents)

    logger.info(
        "Summarising %d document(s) (%d tokens) with style='%s'.",
        len(documents),
        total_tokens,
        style,
    )

    llm = get_llm(temperature=0.0)

    if total_tokens <= threshold:
        return _single_pass_summarize(documents, llm, style_phrase)
    else:
        return _map_reduce_summarize(documents, llm, style_phrase, threshold)


# ---------------------------------------------------------------------------
# Internal strategies
# ---------------------------------------------------------------------------


def _single_pass_summarize(documents: list[Document], llm, style_phrase: str) -> str:
    """Concatenate all document text and summarise in one LLM call."""
    combined_text = "\n\n".join(d.page_content for d in documents if d.page_content.strip())
    chain = _DIRECT_PROMPT | llm | StrOutputParser()
    result: str = chain.invoke({"text": combined_text, "style": style_phrase})
    logger.info("Single-pass summary complete (%d chars).", len(result))
    return result


def _map_reduce_summarize(
    documents: list[Document],
    llm,
    style_phrase: str,
    threshold: int,
) -> str:
    """Chunk documents, summarise each chunk, then reduce to one summary."""
    # Use a smaller chunk size so each map call stays well within context.
    chunk_size = min(threshold, 3_000)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=100)

    logger.info("Map step: summarising %d chunks.", len(chunks))

    map_chain = _MAP_PROMPT | llm | StrOutputParser()
    chunk_summaries: list[str] = []
    for i, chunk in enumerate(chunks):
        try:
            summary: str = map_chain.invoke(
                {"text": chunk.page_content, "style": style_phrase}
            )
            chunk_summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Map step failed for chunk %d: %s", i, exc)
            chunk_summaries.append(chunk.page_content[:500])  # fallback: raw text

    combined = "\n\n".join(
        f"[Section {i + 1}]\n{s}" for i, s in enumerate(chunk_summaries)
    )

    logger.info("Reduce step: collapsing %d chunk summaries.", len(chunk_summaries))
    reduce_chain = _REDUCE_PROMPT | llm | StrOutputParser()
    final: str = reduce_chain.invoke({"text": combined, "style": style_phrase})
    logger.info("Map-Reduce summary complete (%d chars).", len(final))
    return final
