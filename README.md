# Document Intelligence Toolkit

An open-source, enterprise-grade toolkit for intelligent document processing. Ingests PDFs and Word documents, extracts structured data, generates executive summaries, and performs semantic document comparison — powered by LangChain and LLMs, with full offline rule-based fallbacks when no API key is present.

---

## Architecture Overview

```
document-intelligence-toolkit/
│
├── src/
│   ├── document_parsers/          # Ingestion & parsing engine
│   │   ├── pdf_parser.py          # pdfplumber (primary) + pypdf (fallback) → LangChain Documents
│   │   └── docx_parser.py         # python-docx, section-aware → LangChain Documents
│   │
│   ├── intelligence/              # LangChain-powered analysis chains
│   │   ├── extractor.py           # Structured extraction via llm.with_structured_output + Pydantic
│   │   ├── summarizer.py          # Map-Reduce summarization (auto single-pass for short docs)
│   │   └── comparator.py          # deepdiff exact diff + LLM semantic narrative
│   │
│   └── utils/                     # Shared helpers
│       ├── chunker.py             # RecursiveCharacterTextSplitter, token-aware
│       ├── token_counter.py       # tiktoken-based token counting
│       └── llm_factory.py         # Provider-agnostic LLM factory (Anthropic / OpenAI)
│
├── ui/
│   ├── app.py                     # Streamlit two-tab web UI (Analyse + Compare)
│   └── desktop_app.py             # Native Windows desktop app (customtkinter, Anthropic design)
│
├── tests/
│   ├── test_parsers.py            # PDF + DOCX parser unit tests (synthetic fixtures)
│   ├── test_chunker.py            # Chunker + token counter unit tests
│   ├── test_intelligence.py       # Summarizer / extractor / comparator (mocked LLM)
│   ├── test_smoke.py              # End-to-end pipeline smoke test (mocked LLM)
│   └── test_llm_factory.py        # LLM factory — provider/model/temperature tests
│
├── .env.template                  # API key config template — copy to .env
├── .gitignore
├── pyproject.toml                 # Project metadata and dependencies
└── README.md
```

---

## Key Design Decisions

| Concern | Solution |
|---|---|
| Token limit handling | `RecursiveCharacterTextSplitter` + Map-Reduce summarization (auto-selected) |
| Structured extraction | `llm.with_structured_output(DocumentExtraction)` — tool-calling + Pydantic v2 |
| Extraction reliability | Retry-with-feedback loop: validation errors fed back to LLM for correction |
| Semantic diffing | LLM narrative diff with similarity label extraction |
| Exact text diffing | `deepdiff` on document lines before LLM analysis |
| Provider flexibility | `LLM_PROVIDER` / `LLM_MODEL` env vars; `llm_factory.py` is the single switch point |
| Offline / no API key | Rule-based fallbacks in all three intelligence modules (regex, extractive, set-diff) |
| Malformed PDFs | pdfplumber → pypdf fallback; `try/except` guards on every page |
| Large document extraction | Text capped at ~12 000 tokens; use chunker upstream for very large corpora |

---

## Phases

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ Done | Project scaffolding, `pyproject.toml`, `.env.template`, `.gitignore` |
| 2 | ✅ Done | PDF + DOCX parsers, `chunker.py`, `token_counter.py`, parser tests |
| 3 | ✅ Done | `llm_factory.py`, `summarizer.py`, `extractor.py`, `comparator.py`, intelligence tests |
| 4 | ✅ Done | Streamlit two-tab web UI (`ui/app.py`) |
| 5 | ✅ Done | README polish, end-to-end smoke test, CI/CD pipeline |
| 6 | ✅ Done | Retry-with-feedback guardrails, rule-based offline fallbacks |
| 7 | ✅ Done | Native Windows desktop app (`ui/desktop_app.py`) |

---

## Quick Start

```bash
# 1. Clone & enter the project
git clone <repo-url>
cd document-intelligence-toolkit

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies (including dev extras)
pip install -e ".[dev]"

# 4. Configure environment
cp .env.template .env
# Edit .env — add ANTHROPIC_API_KEY or OPENAI_API_KEY (and optionally LLM_PROVIDER / LLM_MODEL)

# 5. Run the test suite (no API key required)
pytest

# 6a. Launch the web UI
streamlit run ui/app.py

# 6b. Or launch the native Windows desktop app (works offline, no API key needed)
python ui/desktop_app.py
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required if `LLM_PROVIDER=anthropic` |
| `OPENAI_API_KEY` | — | Required if `LLM_PROVIDER=openai` |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_MODEL` | `claude-sonnet-4-6` / `gpt-4o` | Override the model name |
| `CHUNK_SIZE` | `1500` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `150` | Token overlap between chunks |
| `MAP_REDUCE_THRESHOLD` | `6000` | Token count above which Map-Reduce is used |

---

## Offline / No API Key Mode

All three intelligence modules automatically fall back to deterministic rule-based processing when no API key is detected:

| Feature | LLM mode | Offline fallback |
|---|---|---|
| Summarisation | Map-Reduce LLM summary | Extractive: first N sentences by style |
| Extraction | `with_structured_output` + retry loop | Regex dates, keyword doc-type, heuristic parties |
| Comparison | LLM semantic narrative | Set-diff of lines + similarity label |

The desktop app (`ui/desktop_app.py`) is designed specifically for offline use.

## LLM Provider Configuration

Set `LLM_PROVIDER` in your `.env` to `anthropic` or `openai`.
The default model is `claude-sonnet-4-6` (Anthropic) or `gpt-4o` (OpenAI).
