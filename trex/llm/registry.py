"""Model registry — maps model strings to the correct LLM client.

Model string formats:
    "gpt-5.4-mini"              → OpenAI
    "gpt-4o"                    → OpenAI
    "claude-haiku-4-5-20251001" → Anthropic
    "claude-sonnet-4-6"         → Anthropic
    "gemini-2.0-flash"          → Gemini
    "ollama/gemma3:27b"         → OpenAI-compatible (localhost:11434)
    "vllm/Qwen3.5-32B"          → OpenAI-compatible (localhost:8000)
    "http://host:port/model"    → OpenAI-compatible (custom endpoint)
"""

from typing import Optional

from .base import LLMClient


# Provider detection rules (checked in order)
_PROVIDER_PREFIXES = [
    ("claude", "anthropic"),
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("gemini", "gemini"),
    ("ollama/", "openai_compat"),
    ("vllm/", "openai_compat"),
    ("http://", "openai_compat"),
    ("https://", "openai_compat"),
]

# Known local API base URLs
_LOCAL_ENDPOINTS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}


def detect_provider(model: str) -> str:
    """Detect the provider from a model string."""
    model_lower = model.lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if model_lower.startswith(prefix):
            return provider
    # Default to OpenAI (most compatible)
    return "openai"


def create_client(model: str, **kwargs) -> LLMClient:
    """Create an LLM client for the given model string.

    Args:
        model: Model identifier (e.g., "gpt-5.4-mini", "claude-haiku-4-5-20251001")
        **kwargs: Extra arguments passed to the client constructor.

    Returns:
        An LLMClient instance configured for the model.
    """
    provider = detect_provider(model)

    if provider == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model=model, **kwargs)

    elif provider == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(model=model, **kwargs)

    elif provider == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient(model=model, **kwargs)

    elif provider == "openai_compat":
        from .openai_client import OpenAIClient
        enable_thinking = kwargs.pop("enable_thinking", True)
        base_url_override = kwargs.pop("base_url", None)

        # Parse "provider/model_name" format
        if "/" in model and not model.startswith("http"):
            prefix, actual_model = model.split("/", 1)
            base_url = base_url_override or _LOCAL_ENDPOINTS.get(prefix, f"http://localhost:8000/v1")
            return OpenAIClient(
                model=actual_model,
                base_url=base_url,
                api_key="not-needed",  # Local models don't need keys
                enable_thinking=enable_thinking,
                **kwargs,
            )
        elif model.startswith("http"):
            # Custom endpoint: "http://host:port/model_name"
            # Split URL from model name at last /
            parts = model.rsplit("/", 1)
            base_url = base_url_override or (parts[0] + "/v1" if len(parts) > 1 else model)
            actual_model = parts[1] if len(parts) > 1 else "default"
            return OpenAIClient(
                model=actual_model,
                base_url=base_url,
                api_key="not-needed",
                enable_thinking=enable_thinking,
                **kwargs,
            )

    raise ValueError(f"Unknown provider for model: {model}")


def list_providers() -> dict:
    """List supported providers and example model strings."""
    return {
        "anthropic": {
            "examples": ["claude-haiku-4-5-20251001", "claude-sonnet-4-6", "claude-opus-4-6"],
            "env_key": "ANTHROPIC_API_KEY",
        },
        "openai": {
            "examples": ["gpt-5.4-mini", "gpt-4o", "gpt-4o-mini", "o4-mini"],
            "env_key": "OPENAI_API_KEY",
        },
        "gemini": {
            "examples": ["gemini-2.0-flash", "gemini-2.5-pro"],
            "env_key": "GEMINI_API_KEY",
        },
        "openai_compat": {
            "examples": ["ollama/gemma3:27b", "vllm/Qwen3.5-32B"],
            "env_key": "None (local)",
        },
    }
