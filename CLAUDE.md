# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (including dev)
pip install -e ".[dev]"

# Run full test suite (no API key required — all LLM calls are mocked)
pytest

# Run a single test file
pytest tests/test_intelligence.py -v

# Run a single test by name
pytest tests/test_intelligence.py::TestRetryWithFeedback::test_retries_on_quality_failure_then_succeeds -v

# Lint
ruff check src/ tests/ ui/

# Format check
ruff format --check src/ tests/ ui/

# Auto-fix formatting
ruff format src/ tests/ ui/

# Launch web UI (requires API key in .env)
streamlit run ui/app.py

# Launch native Windows desktop app (works offline, no API key needed)
python ui/desktop_app.py
```

## Architecture

The pipeline is: **parse → chunk → analyse (summarise / extract / compare)**.

### Dual-mode intelligence

Every intelligence module checks `has_api_key()` from `src/utils/llm_factory.py` at call time and branches:
- **LLM mode** — LangChain chains via `get_llm()` (Anthropic or OpenAI, switched by `LLM_PROVIDER` env var)
- **Offline fallback** — deterministic rule-based functions (`_extractive_summarize`, `_rule_based_extract`, `_rule_based_diff_summary`) that require no network access

Never call `get_llm()` without first checking `has_api_key()` if the operation should be optional.

### Extractor guardrail loop

`extract_structured_data()` in `src/intelligence/extractor.py` runs a retry-with-feedback loop:
1. First attempt via `_EXTRACTION_PROMPT | structured_llm`
2. `validate_extraction()` checks quality (empty title, short summary, empty topics)
3. On failure, issues are formatted and sent back via `_RETRY_PROMPT` (up to `max_retries` times)
4. Exhausted retries → returns best-effort result with a warning log (never raises)

### Summariser strategy selection

`summarize_documents()` auto-selects between single-pass and Map-Reduce based on total token count vs `MAP_REDUCE_THRESHOLD` (default 6 000, overridable via env var). Map-Reduce chunk size is capped at `min(threshold, 3_000)` to keep each map call within context.

### LLM factory

`src/utils/llm_factory.py` is the **only** place that imports `ChatAnthropic` / `ChatOpenAI`. All other modules call `get_llm()`. `has_api_key()` checks the relevant key for the configured provider without instantiating the client.

### Document parsers

Both parsers return `list[Document]` (LangChain). PDF uses pdfplumber primary / pypdf fallback per page. DOCX splits by heading into sections (not pages). Metadata keys differ: PDF → `total_pages`, DOCX → `total_sections` — downstream code must handle both.

### Testing conventions

All LLM calls are mocked via `unittest.mock.patch("src.intelligence.<module>.get_llm")`. No API key is needed to run any test. Synthetic DOCX fixtures are built with `python-docx` in `tmp_path`; synthetic PDFs with raw PDF bytes. Never use real files in tests.

## Key constraints

- `pyproject.toml` build backend must be `setuptools.build_meta` (not `setuptools.backends.legacy`) — Python 3.14 compatibility.
- `ruff` line length is 100; target is Python 3.10+.
- Type hints required on all public functions. Internal helpers use `ChatModel` (from `llm_factory`) for LLM parameters, not bare `Any`.
- Broad `except Exception` is permitted only with `# noqa: BLE001` and a logger call — never silently swallowed.
