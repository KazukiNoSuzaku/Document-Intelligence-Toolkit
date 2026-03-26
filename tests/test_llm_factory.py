"""Unit tests for LLM factory.

Verifies provider selection, model defaults, and temperature passthrough
without making any real API calls.

Run with:  pytest tests/test_llm_factory.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from src.utils.llm_factory import _ANTHROPIC_DEFAULT, _OPENAI_DEFAULT, get_llm


class TestGetLlmProvider:
    @patch.dict("os.environ", {"LLM_PROVIDER": "anthropic"}, clear=False)
    def test_anthropic_provider_returns_chat_anthropic(self) -> None:
        llm = get_llm()
        assert isinstance(llm, ChatAnthropic)

    @patch.dict("os.environ", {"LLM_PROVIDER": "openai"}, clear=False)
    def test_openai_provider_returns_chat_openai(self) -> None:
        llm = get_llm()
        assert isinstance(llm, ChatOpenAI)

    @patch.dict("os.environ", {}, clear=False)
    def test_defaults_to_anthropic_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            # Remove LLM_PROVIDER if present
            import os
            os.environ.pop("LLM_PROVIDER", None)
            llm = get_llm()
            assert isinstance(llm, ChatAnthropic)

    @patch.dict("os.environ", {"LLM_PROVIDER": "  Anthropic  "}, clear=False)
    def test_strips_and_lowercases_provider(self) -> None:
        llm = get_llm()
        assert isinstance(llm, ChatAnthropic)

    @patch.dict("os.environ", {"LLM_PROVIDER": "unknown_provider"}, clear=False)
    def test_unknown_provider_falls_back_to_anthropic(self) -> None:
        llm = get_llm()
        assert isinstance(llm, ChatAnthropic)


class TestGetLlmModel:
    @patch.dict("os.environ", {"LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-haiku-4-5-20251001"}, clear=False)
    def test_anthropic_model_override(self) -> None:
        llm = get_llm()
        assert isinstance(llm, ChatAnthropic)
        assert llm.model == "claude-haiku-4-5-20251001"

    @patch.dict("os.environ", {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o-mini"}, clear=False)
    def test_openai_model_override(self) -> None:
        llm = get_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == "gpt-4o-mini"

    @patch.dict("os.environ", {"LLM_PROVIDER": "anthropic"}, clear=False)
    def test_anthropic_default_model(self) -> None:
        import os
        os.environ.pop("LLM_MODEL", None)
        llm = get_llm()
        assert isinstance(llm, ChatAnthropic)
        assert llm.model == _ANTHROPIC_DEFAULT

    @patch.dict("os.environ", {"LLM_PROVIDER": "openai"}, clear=False)
    def test_openai_default_model(self) -> None:
        import os
        os.environ.pop("LLM_MODEL", None)
        llm = get_llm()
        assert isinstance(llm, ChatOpenAI)
        assert llm.model_name == _OPENAI_DEFAULT


class TestGetLlmTemperature:
    @patch.dict("os.environ", {"LLM_PROVIDER": "anthropic"}, clear=False)
    def test_temperature_passed_to_anthropic(self) -> None:
        llm = get_llm(temperature=0.7)
        assert isinstance(llm, ChatAnthropic)
        assert llm.temperature == 0.7

    @patch.dict("os.environ", {"LLM_PROVIDER": "openai"}, clear=False)
    def test_temperature_passed_to_openai(self) -> None:
        llm = get_llm(temperature=0.5)
        assert isinstance(llm, ChatOpenAI)
        assert llm.temperature == 0.5

    @patch.dict("os.environ", {"LLM_PROVIDER": "anthropic"}, clear=False)
    def test_default_temperature_is_zero(self) -> None:
        llm = get_llm()
        assert llm.temperature == 0.0
