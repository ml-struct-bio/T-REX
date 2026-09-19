"""Anthropic (Claude) LLM client."""

import json
from typing import List, Optional

import anthropic

from .base import LLMClient, LLMResponse, ToolCall


class AnthropicClient(LLMClient):
    """Claude via Anthropic SDK. Tools use native format (no conversion needed)."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", **kwargs):
        super().__init__(model)
        self.client = anthropic.Anthropic(**kwargs)

    def chat(
        self,
        messages: List[dict],
        system: str = "",
        tools: Optional[List[dict]] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = self.client.messages.create(**kwargs)

        # Parse response
        text_parts = []
        tool_calls = []
        for block in resp.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=resp.model,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
        )

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": result,
        }

    def format_assistant_message(self, response: LLMResponse) -> dict:
        content = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
        return {"role": "assistant", "content": content}
