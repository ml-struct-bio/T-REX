"""Abstract base class for LLM clients with tool use."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """A normalized tool call from any LLM provider."""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    stop_reason: str = ""  # "end_turn", "tool_use", "length"
    model: str = ""
    usage: Optional[dict] = None  # {"input_tokens": ..., "output_tokens": ...}
    thinking: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# Canonical tool schema format (Anthropic-style, converted per provider)
# {
#     "name": "tool_name",
#     "description": "what it does",
#     "input_schema": { "type": "object", "properties": {...}, "required": [...] }
# }


class LLMClient(ABC):
    """Abstract LLM client supporting chat + tool use."""

    def __init__(self, model: str, **kwargs):
        self.model = model

    @abstractmethod
    def chat(
        self,
        messages: List[dict],
        system: str = "",
        tools: Optional[List[dict]] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Send a chat request with optional tool definitions.

        Args:
            messages: Conversation history in canonical format:
                [{"role": "user"|"assistant"|"tool", "content": ...}]
            system: System prompt string.
            tools: Tool definitions in canonical (Anthropic) format.
            max_tokens: Maximum tokens in response.
            temperature: Optional provider sampling temperature.

        Returns:
            Normalized LLMResponse with text and/or tool calls.
        """
        ...

    @abstractmethod
    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        """Format a tool result for inclusion in the next message.

        Each provider expects tool results in a different format.
        This method returns a message dict ready to append to the conversation.
        """
        ...

    @abstractmethod
    def format_assistant_message(self, response: LLMResponse) -> dict:
        """Format the assistant's response for conversation history.

        Converts the normalized response back into provider-specific format
        for inclusion in the messages list.
        """
        ...

    @property
    def provider(self) -> str:
        return self.__class__.__name__.replace("Client", "").lower()
