"""Google Gemini LLM client."""

import json
import uuid
from typing import List, Optional

from google import genai
from google.genai import types

from .base import LLMClient, LLMResponse, ToolCall


def _convert_tools_to_gemini(tools: List[dict]) -> List[types.Tool]:
    """Convert canonical tool format to Gemini function declarations."""
    declarations = []
    for tool in tools:
        schema = tool.get("input_schema", {"type": "object", "properties": {}})
        declarations.append(types.FunctionDeclaration(
            name=tool["name"],
            description=tool.get("description", ""),
            parameters=schema,
        ))
    return [types.Tool(function_declarations=declarations)]


def _convert_messages_to_gemini(
    messages: List[dict],
) -> List[types.Content]:
    """Convert canonical messages to Gemini Content objects."""
    contents = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            if isinstance(content, list):
                # Could be tool results
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        parts.append(types.Part.from_function_response(
                            name=block.get("_tool_name", "unknown"),
                            response=json.loads(block["content"]) if isinstance(block["content"], str) else block["content"],
                        ))
                    elif isinstance(block, str):
                        parts.append(types.Part.from_text(text=block))
                if parts:
                    contents.append(types.Content(role="user", parts=parts))
            else:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=content)],
                ))

        elif role == "assistant":
            parts = []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text" and block.get("text"):
                            parts.append(types.Part.from_text(text=block["text"]))
                        elif block.get("type") == "tool_use":
                            parts.append(types.Part.from_function_call(
                                name=block["name"],
                                args=block["input"],
                            ))
            elif isinstance(content, str):
                parts.append(types.Part.from_text(text=content))
            if parts:
                contents.append(types.Content(role="model", parts=parts))

    return contents


class GeminiClient(LLMClient):
    """Gemini via Google GenAI SDK."""

    def __init__(self, model: str = "gemini-2.0-flash", **kwargs):
        super().__init__(model)
        self.client = genai.Client(**kwargs)

    def chat(
        self,
        messages: List[dict],
        system: str = "",
        tools: Optional[List[dict]] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        gemini_contents = _convert_messages_to_gemini(messages)

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
        )
        if temperature is not None:
            config.temperature = float(temperature)
        if system:
            config.system_instruction = system
        if tools:
            config.tools = _convert_tools_to_gemini(tools)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )

        # Parse response
        text_parts = []
        tool_calls = []

        if resp.candidates and resp.candidates[0].content:
            for part in resp.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    fc = part.function_call
                    tool_calls.append(ToolCall(
                        id=f"gemini_{uuid.uuid4().hex[:8]}",
                        name=fc.name,
                        arguments=dict(fc.args) if fc.args else {},
                    ))

        usage = None
        if resp.usage_metadata:
            usage = {
                "input_tokens": resp.usage_metadata.prompt_token_count,
                "output_tokens": resp.usage_metadata.candidates_token_count,
            }

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            model=self.model,
            usage=usage,
        )

    def format_tool_result(self, tool_call: ToolCall, result: str) -> dict:
        # Store in canonical format with extra _tool_name for Gemini conversion
        return {
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "_tool_name": tool_call.name,
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
