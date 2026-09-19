"""Network-free characterization of the existing generic LLM adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trex.llm import openai_client
from trex.llm.registry import create_client


@pytest.fixture
def fake_sdk(monkeypatch):
    constructor_calls = []
    request_calls = []

    def complete(**kwargs):
        request_calls.append(kwargs)
        return SimpleNamespace(
            model="server-reported-model",
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="test response", tool_calls=None, reasoning=None,
            ))],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        )

    def construct(**kwargs):
        constructor_calls.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=complete),
        ))

    monkeypatch.setattr(openai_client, "OpenAI", construct)
    return constructor_calls, request_calls


@pytest.mark.parametrize("model, endpoint, served_model, expected_endpoint", [
    ("vllm/test-model", None, "test-model", "http://localhost:8000/v1"),
    ("vllm/org/test-model", None, "org/test-model", "http://localhost:8000/v1"),
    ("vllm/test-model", "http://example.invalid:9000/v1",
     "test-model", "http://example.invalid:9000/v1"),
])
def test_model_routing_preserves_served_name_and_endpoint(
    fake_sdk, model, endpoint, served_model, expected_endpoint,
) -> None:
    constructors, requests = fake_sdk
    kwargs = {} if endpoint is None else {"base_url": endpoint}

    client = create_client(model, **kwargs)

    assert client.model == served_model
    assert client.enable_thinking is True
    assert constructors == [{"base_url": expected_endpoint, "api_key": "not-needed"}]
    assert requests == []  # Construction does not submit a model request.


@pytest.mark.parametrize("thinking, requested_tokens, expected_tokens", [
    (False, 17, 17),
    (True, 17, 8192),
    (True, 9000, 9000),
])
def test_request_defaults_and_response_accounting_are_unchanged(
    fake_sdk, thinking, requested_tokens, expected_tokens,
) -> None:
    _, requests = fake_sdk
    messages = [{"role": "user", "content": "Return a test response."}]
    client = create_client("vllm/test-model", enable_thinking=thinking)

    response = client.chat(
        messages, system="Generic software fixture.",
        max_tokens=requested_tokens, temperature=0.0,
    )

    expected = {
        "model": "test-model", "max_completion_tokens": expected_tokens,
        "messages": [
            {"role": "system", "content": "Generic software fixture."},
            {"role": "user", "content": "Return a test response."},
        ],
        "temperature": 0.0,
    }
    if thinking:
        expected["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    assert requests == [expected]
    assert messages == [{"role": "user", "content": "Return a test response."}]
    assert response.text == "test response"
    assert response.model == "server-reported-model"
    assert response.usage == {"input_tokens": 4, "output_tokens": 2}
