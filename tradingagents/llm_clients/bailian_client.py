import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


# Bailian Standard API Configuration
BAILIAN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Qwen 3.5 series (latest generation - July 2025)
QWEN_35_SERIES = [
    "qwen3.5",
    "qwen3.5-turbo",
    "qwen3.5-plus",
    "qwen3.5-max",
    "qwen3.5-coder",
]

# Qwen 3 series (previous generation - April 2025)
QWEN_3_SERIES = [
    "qwen3",
    "qwen3-turbo",
    "qwen3-plus",
]

# Legacy Qwen series
LEGACY_QWEN_SERIES = [
    "qwen-max",
    "qwen-plus",
    "qwen-turbo",
    "qwen-long",
    "qwq-plus",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
]


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output for Bailian."""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


_PASSTHROUGH_KWARGS = (
    "timeout",
    "max_retries",
    "api_key",
    "callbacks",
    "http_client",
    "http_async_client",
)


class BailianClient(BaseLLMClient):
    """Client for Alibaba Cloud Bailian Standard API (pay-per-use).
    
    Configuration:
        - API Key: sk-xxxxx (from Bailian console)
        - Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
        - Pricing: Pay-per-use based on token consumption
    
    Usage:
        # Direct instantiation
        client = BailianClient(
            model="qwen3.5-max",
            api_key="sk-xxxxx"
        )
        
        # Or via environment variable
        export BAILIAN_API_KEY=sk-xxxxx
        client = BailianClient(model="qwen3.5-max")
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance for Bailian Standard API."""
        llm_kwargs = {
            "model": self.model,
            "base_url": self.base_url or BAILIAN_BASE_URL,
        }

        # Get API key from kwargs or environment
        api_key = self.kwargs.get("api_key") or os.environ.get("BAILIAN_API_KEY")
        if api_key:
            llm_kwargs["api_key"] = api_key

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Bailian uses OpenAI-compatible Chat Completions API
        llm_kwargs["use_responses_api"] = False

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Bailian Standard API."""
        return validate_model("bailian", self.model)
