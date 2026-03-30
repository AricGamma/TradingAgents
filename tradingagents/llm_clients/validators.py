"""Model name validators for each provider.

Only validates model names - does NOT enforce limits.
Let LLM providers use their own defaults for unspecified params.
"""

VALID_MODELS = {
    "openai": [
        # GPT-5 series
        "gpt-5.4-pro",
        "gpt-5.4",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        # GPT-4.1 series
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ],
    "anthropic": [
        # Claude 4.6 series (latest)
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        # Claude 4.5 series
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
    ],
    "google": [
        # Gemini 3.1 series (preview)
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite-preview",
        # Gemini 3 series (preview)
        "gemini-3-flash-preview",
        # Gemini 2.5 series
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    "xai": [
        # Grok 4.1 series
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
        # Grok 4 series
        "grok-4-0709",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
    ],
    "bailian": [
        # Qwen (通义千问) series
        "qwen-max",
        "qwen-plus",
        "qwen-turbo",
        "qwen-long",
        "qwq-plus",
        "qwen2.5-72b-instruct",
        "qwen2.5-32b-instruct",
        # Qwen 3.5 series (latest generation)
        "qwen3.5",
        "qwen3.5-turbo",
        "qwen3.5-plus",
        "qwen3.5-max",
        "qwen3.5-coder",
        # Qwen 3 series
        "qwen3",
        "qwen3-turbo",
        "qwen3-plus",
        # Coding Plan 代码模型
        "qwen3-coder-next",
        "qwen3-coder-plus",
        "qwen3-max-2026-01-23",
        "glm-5",
        "glm-4.7",
        "kimi-k2.5",
        "MiniMax-M2.5",
    ],
}


def validate_model(provider: str, model: str) -> bool:
    """Check if model name is valid for the given provider.

    For ollama, openrouter, bailian, bailian_coding_plan - any model is accepted.
    """
    provider_lower = provider.lower()

    if provider_lower in ("ollama", "openrouter", "bailian", "bailian_coding_plan"):
        return True

    if provider_lower not in VALID_MODELS:
        return True

    return model in VALID_MODELS[provider_lower]
