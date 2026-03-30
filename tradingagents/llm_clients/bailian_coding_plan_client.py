import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


# Coding Plan Configuration
CODING_PLAN_BASE_URL = "https://coding.dashscope.aliyuncs.com/v1"
CODING_PLAN_API_KEY_PREFIX = "sk-sp-"

# Coding Plan supported models
CODING_PLAN_MODELS = [
    "qwen3-coder-next",
    "qwen3-coder-plus",
    "qwen3.5-coder",
    "qwen3.5-plus",
    "qwen3.5-max",
    "glm-5",
    "glm-4.7",
    "kimi-k2.5",
    "MiniMax-M2.5",
]


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output for Coding Plan."""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


_PASSTHROUGH_KWARGS = (
    "timeout",
    "max_retries",
    "api_key",
    "callbacks",
    "http_client",
    "http_async_client",
    "stream_usage",
)


class BailianCodingPlanClient(BaseLLMClient):
    """Client for Alibaba Cloud Bailian Coding Plan.
    
    Coding Plan is a subscription-based service ($50/month) for coding tasks.
    
    Configuration:
        - Exclusive API Key: sk-sp-xxxxx (from Coding Plan page)
        - Exclusive Base URL: https://coding.dashscope.aliyuncs.com/v1
        - Subscription: https://modelstudio.console.aliyuncs.com
    
    Usage:
        # Direct instantiation
        client = BailianCodingPlanClient(
            model="qwen3-coder-plus",
            api_key="sk-sp-xxxxx"
        )
        
        # Or via environment variable
        export BAILIAN_CODING_PLAN_API_KEY=sk-sp-xxxxx
        client = BailianCodingPlanClient(model="qwen3-coder-plus")
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance for Coding Plan."""
        llm_kwargs = {
            "model": self.model,
            "base_url": self.base_url or CODING_PLAN_BASE_URL,
        }

        # Get API key from kwargs or environment
        api_key = self.kwargs.get("api_key") or os.environ.get("BAILIAN_CODING_PLAN_API_KEY")
        if api_key:
            llm_kwargs["api_key"] = api_key

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Coding Plan uses OpenAI-compatible Chat Completions API
        llm_kwargs["use_responses_api"] = False

        return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Coding Plan."""
        return validate_model("bailian_coding_plan", self.model)
