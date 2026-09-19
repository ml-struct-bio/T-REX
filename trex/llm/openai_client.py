"""OpenAI (GPT) LLM client. Also works with any OpenAI-compatible API."""

import json
import uuid
from typing import List, Optional

from openai import OpenAI

from .base import LLMClient, LLMResponse, ToolCall


def _convert_tools_to_openai(tools: List[dict]) -> List[dict]:
    """Convert canonical (Anthropic) tool format to OpenAI format.

    Anthropic: {"name", "description", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name", "description", "parameters": {...}}}
    """
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            },
        })
    return result


def _convert_messages_to_openai(messages: List[dict], system: str = "") -> List[dict]:
    """Convert canonical messages to OpenAI format.

    Key differences:
    - System message is a regular message with role="system"
    - Tool results use role="tool" with tool_call_id
    - Assistant tool_use blocks become tool_calls array
    """
    result = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            # Check if content has tool_use blocks (Anthropic format)
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block["id"],
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block["input"]),
                                },
                            })
                oai_msg = {"role": "assistant"}
                if text_parts:
                    oai_msg["content"] = "\n".join(text_parts)
                else:
                    oai_msg["content"] = None if tool_calls else ""
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)
            else:
                result.append({"role": "assistant", "content": content})

        elif role == "user":
            # Check for tool_result blocks (Anthropic format)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        })
                    else:
                        result.append({"role": "user", "content": str(block)})
            else:
                result.append({"role": "user", "content": content})

        else:
            result.append(msg)

    return result


class OpenAIClient(LLMClient):
    """GPT via OpenAI SDK. Also works with any OpenAI-compatible API.

    When ``enable_thinking`` is True, sends the Qwen/vLLM
    ``chat_template_kwargs.enable_thinking`` flag and stores any returned
    reasoning text on ``LLMResponse.thinking``.
    """

    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        enable_thinking: bool = False,
        **kwargs,
    ):
        super().__init__(model)
        self.enable_thinking = enable_thinking
        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        client_kwargs.update(kwargs)
        self.client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: List[dict],
        system: str = "",
        tools: Optional[List[dict]] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        oai_messages = _convert_messages_to_openai(messages, system)
        effective_max = max(max_tokens, 8192) if self.enable_thinking else max_tokens

        kwargs = {
            "model": self.model,
            "max_completion_tokens": effective_max,
            "messages": oai_messages,
        }
        if temperature is not None:
            kwargs["temperature"] = float(temperature)
        if tools:
            kwargs["tools"] = _convert_tools_to_openai(tools)
        if self.enable_thinking:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}

        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        thinking_text = getattr(choice.message, "reasoning", None) or ""

        # Parse tool calls
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=resp.model,
            usage={
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            } if resp.usage else None,
            thinking=thinking_text,
        )

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        # Return in canonical (Anthropic) format — conversion happens in chat()
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "content": result,
        }

    def format_assistant_message(self, response: LLMResponse) -> dict:
        # Store in canonical (Anthropic) format — conversion happens in chat()
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
