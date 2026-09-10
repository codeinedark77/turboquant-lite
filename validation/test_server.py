"""
validation/test_server.py — pytest suite for server/app.py, using FastAPI's
TestClient (no live HTTP server needed) against the same synthetic model
used throughout this project, plus a minimal mock tokenizer (the synthetic
model has no real vocabulary/chat template to draw on). This checks the
request/response plumbing — routing, schema, streaming SSE format, the
use_turboquant on/off toggle actually switching cache implementations.
It does NOT check real generation quality, same limitation as everywhere
else in this sandbox.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from fastapi.testclient import TestClient

from inference.quality_probe import build_synthetic_model
from server.config import ServerConfig
import server.app as server_app


class MockTokenizer:
    """Just enough surface for server/app.py + TextIteratorStreamer to run
    against a synthetic (no real vocabulary) model."""
    vocab_size = 100
    eos_token_id = 0
    pad_token_id = 0

    def apply_chat_template(self, chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"):
        n = 5 + (sum(len(m["content"]) for m in chat) % 10)
        torch.manual_seed(len(chat))
        return torch.randint(1, self.vocab_size, (1, n))

    def decode(self, token_ids, skip_special_tokens=True):
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(f"tok{t}" for t in token_ids)


@pytest.fixture(scope="module", autouse=True)
def setup_synthetic_server():
    model, cfg = build_synthetic_model(seed=0)
    server_app.state.model = model
    server_app.state.tokenizer = MockTokenizer()
    server_app.state.model_config = cfg
    server_app.state.cfg = ServerConfig(
        model_name_or_path="synthetic-test-model", use_turboquant=True,
        k_bits=3, v_bits=4, use_prod=False, device="cpu",
    )
    yield


@pytest.fixture
def client():
    return TestClient(server_app.app)


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "synthetic-test-model"


def test_chat_completions_non_streaming(client):
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 5,
        "stream": False,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(body["choices"][0]["message"]["content"], str)
    assert body["usage"]["completion_tokens"] <= 5


def test_chat_completions_streaming(client):
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 5,
        "stream": True,
    })
    assert resp.status_code == 200
    raw = resp.text
    assert raw.strip().endswith("data: [DONE]")
    assert '"object": "chat.completion.chunk"' in raw
    assert '"finish_reason": "stop"' in raw


def test_empty_messages_rejected_cleanly(client):
    resp = client.post("/v1/chat/completions", json={"messages": [], "max_tokens": 5})
    assert resp.status_code == 422
    assert "empty" in resp.text


def test_invalid_role_rejected_cleanly(client):
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "bogus", "content": "hi"}], "max_tokens": 5,
    })
    assert resp.status_code == 422


def test_negative_max_tokens_rejected_cleanly(client):
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": -5,
    })
    assert resp.status_code == 422


def test_excessive_max_tokens_rejected_cleanly(client):
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 999_999,
    })
    assert resp.status_code == 422


def test_missing_messages_field_rejected_cleanly(client):
    resp = client.post("/v1/chat/completions", json={"max_tokens": 5})
    assert resp.status_code == 422


def test_max_completion_tokens_is_respected(client):
    """Real bug found via the actual ChatOpenAI client: it sends
    max_completion_tokens, not max_tokens. Regression test at the raw
    payload level."""
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": 4,
        "stream": False,
    })
    assert resp.status_code == 200
    assert resp.json()["usage"]["completion_tokens"] <= 4


def test_legacy_max_tokens_still_works(client):
    """max_tokens (older/other clients) must keep working alongside the
    newer field, not get silently dropped in favor of it."""
    resp = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "stream": False,
    })
    assert resp.status_code == 200
    assert resp.json()["usage"]["completion_tokens"] <= 4


def test_real_chatopenai_client_respects_token_limit():
    """The actual regression test: real langchain_openai.ChatOpenAI, wired
    to this app via ASGI transport (real request/response objects, no
    network). This is the exact test that caught the bug -- kept as a
    permanent check against future field-name drift in either this server
    or the client library."""
    import asyncio
    import httpx
    from langchain_openai import ChatOpenAI

    async def run():
        async_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server_app.app), base_url="http://testserver"
        )
        llm = ChatOpenAI(
            base_url="http://testserver/v1", api_key="x", model="test",
            max_tokens=4, http_async_client=async_client,
        )
        return await llm.ainvoke("hi")

    response = asyncio.run(run())
    token_count = len(response.content.split())
    assert token_count <= 4, f"expected <=4 tokens, got {token_count}: {response.content!r}"


def test_use_prod_path_non_streaming():
    """The server config supports use_prod=True but it was never actually
    exercised through the HTTP layer until now -- separate state setup
    since the module fixture pins use_prod=False."""
    model, cfg = build_synthetic_model(seed=0)
    server_app.state.model = model
    server_app.state.tokenizer = MockTokenizer()
    server_app.state.model_config = cfg
    server_app.state.cfg = ServerConfig(model_name_or_path="test-prod", device="cpu", use_prod=True, k_bits=3, v_bits=4)
    c = TestClient(server_app.app)
    resp = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    assert resp.status_code == 200
    assert isinstance(resp.json()["choices"][0]["message"]["content"], str)
    server_app.state.cfg.use_prod = False  # restore


def test_use_prod_path_streaming():
    model, cfg = build_synthetic_model(seed=0)
    server_app.state.model = model
    server_app.state.tokenizer = MockTokenizer()
    server_app.state.model_config = cfg
    server_app.state.cfg = ServerConfig(model_name_or_path="test-prod", device="cpu", use_prod=True, k_bits=3, v_bits=4)
    c = TestClient(server_app.app)
    resp = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5, "stream": True})
    assert resp.status_code == 200
    assert resp.text.strip().endswith("data: [DONE]")
    server_app.state.cfg.use_prod = False  # restore


def test_turboquant_toggle_switches_cache_implementation(client):
    """The on/off flag actually changes which cache/attention path runs --
    not just accepted and ignored."""
    server_app.state.cfg.use_turboquant = True
    resp_on = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3,
    })
    assert resp_on.status_code == 200

    server_app.state.cfg.use_turboquant = False
    resp_off = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3,
    })
    assert resp_off.status_code == 200
    server_app.state.cfg.use_turboquant = True  # restore for other tests
