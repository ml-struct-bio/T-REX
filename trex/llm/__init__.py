"""Multi-provider LLM abstraction for the agent loop.

Supports: Claude (Anthropic), GPT (OpenAI), Gemini (Google),
and any OpenAI-compatible API (vLLM, Ollama, etc.)

Usage:
    from trex.llm import create_client

    client = create_client("gpt-5.4-mini")        # OpenAI
    client = create_client("claude-haiku-4-5")     # Anthropic
    client = create_client("gemini-2.0-flash")     # Google
    client = create_client("ollama/gemma3:27b")    # Local via OpenAI-compat
"""

from .base import LLMClient, LLMResponse, ToolCall
from .registry import create_client, list_providers

__all__ = ["LLMClient", "LLMResponse", "ToolCall", "create_client", "list_providers"]
