"""LLM factory — returns a configured LangChain chat model.

Reads ``LLM_PROVIDER`` (default: ``anthropic``) and ``LLM_MODEL`` from the
environment so the rest of the pipeline stays provider-agnostic.
"""

from __future__ import annotations

import os
from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

ChatModel = Union[ChatAnthropic, ChatOpenAI]

_ANTHROPIC_DEFAULT = "claude-sonnet-4-6"
_OPENAI_DEFAULT = "gpt-4o"


def has_api_key() -> bool:
    """Return True if an API key for the configured provider is present.

    Used by intelligence modules to decide whether to use LLM-powered or
    rule-based fallback processing.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY", "").strip())
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


def get_llm(temperature: float = 0.0) -> ChatModel:
    """Return a configured chat model based on environment variables.

    Environment variables:
        LLM_PROVIDER: ``anthropic`` (default) or ``openai``.
        LLM_MODEL: Model name override. Defaults to ``claude-sonnet-4-6``
            for Anthropic and ``gpt-4o`` for OpenAI.

    Args:
        temperature: Sampling temperature passed to the model.

    Returns:
        A LangChain ``BaseChatModel`` instance ready for invocation.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower().strip()

    if provider == "openai":
        model = os.getenv("LLM_MODEL", _OPENAI_DEFAULT)
        return ChatOpenAI(model=model, temperature=temperature)

    # Default: Anthropic
    model = os.getenv("LLM_MODEL", _ANTHROPIC_DEFAULT)
    return ChatAnthropic(model=model, temperature=temperature)  # type: ignore[call-arg]
