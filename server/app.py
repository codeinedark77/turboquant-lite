"""
server/app.py — Phase 5: the pluggable deliverable.

An OpenAI-compatible /v1/chat/completions server. Point any agent's
base_url at this and swap between it and a cloud API with zero code changes
on the agent side — that's the whole point of building this as a separate,
attachable project rather than baking KV-cache compression into any one
agent directly.

Uses Phase 2's plain MSE-K/V cache by default (inference/kv_cache.py) —
the config that actually won every head-to-head test this session, not
Phase 2b's theoretically-motivated but empirically-unproven Prod path (see
server/config.py, ARCHITECTURE.md Phase 2b status). Set
ServerConfig.use_prod=True to try the other path; nothing here hides that
knob or pretends the choice is settled.

Real usage (needs your actual GPU + real weights, not this dev sandbox):
    uvicorn server.app:app --host 0.0.0.0 --port 8000
    # then point any OpenAI-client-compatible agent at http://localhost:8000/v1

This file's request/response plumbing IS tested here (FastAPI TestClient +
the same synthetic model used throughout this project, see
validation/test_server.py) — what's NOT tested here is real weights on a
real GPU, for the same reason as everywhere else in this project.
"""
from __future__ import annotations
import sys
import os
import time
import uuid
import json
import threading
from typing import Literal, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from server.config import ServerConfig


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "turboquant-lite"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False

    @property
    def resolved_max_tokens(self) -> int:
        """`max_completion_tokens` is what real OpenAI-client libraries send
        now (found via ChatOpenAI, not documented anywhere obvious up
        front) -- `max_tokens` is kept for older/other clients. If both
        arrive, `max_completion_tokens` wins, matching OpenAI's own
        migration precedent."""
        return self.max_completion_tokens or self.max_tokens or 256

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages must not be empty")
        return v

    @field_validator("max_tokens", "max_completion_tokens")
    @classmethod
    def max_tokens_sane(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens/max_completion_tokens must be positive")
        if v is not None and v > 32768:
            raise ValueError("max_tokens/max_completion_tokens exceeds server limit of 32768")
        return v


class AppState:
    """Holds the loaded model/tokenizer/config. Populated by load_real_model()
    for real use, or injected directly in tests (see test_server.py) so the
    HTTP/generation plumbing is checkable without a GPU or real weights."""
    model = None
    tokenizer = None
    cfg: ServerConfig = ServerConfig()
    model_config = None  # the HF config object -- num_hidden_layers, head_dim, etc.


state = AppState()
app = FastAPI(title="turboquant-lite")


def load_real_model(cfg: ServerConfig):
    """Real startup path -- run this on your actual hardware. Not exercised
    in the dev sandbox (no GPU, no network path to gated weights there)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

    state.cfg = cfg
    state.model_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    state.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    state.model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, quantization_config=bnb_config, device_map=cfg.device
    )
    state.model.eval()


def _build_cache():
    """None -> model's own default DynamicCache. Centralized here so the
    on/off toggle in ServerConfig is the only place this decision is made."""
    if not state.cfg.use_turboquant:
        state.model.set_attn_implementation("eager")
        return None
    if state.cfg.use_prod:
        from inference.attention_hook import build_turboquant_prod_cache, register_turboquant_prod_attention
        cache, layers = build_turboquant_prod_cache(
            state.model_config.num_hidden_layers, state.model_config.head_dim,
            state.cfg.k_bits, state.cfg.v_bits,
        )
        impl = register_turboquant_prod_attention(layers, name=f"server_prod_{uuid.uuid4().hex[:8]}")
        state.model.set_attn_implementation(impl)
        return cache
    else:
        from inference.kv_cache import build_turboquant_cache
        state.model.set_attn_implementation("eager")
        return build_turboquant_cache(
            state.model_config.num_hidden_layers, state.model_config.head_dim,
            state.cfg.k_bits, state.cfg.v_bits,
        )


def _prompt_ids(messages: list[ChatMessage]):
    chat = [{"role": m.role, "content": m.content} for m in messages]
    device = next(state.model.parameters()).device
    return state.tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)


@app.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": state.cfg.model_name_or_path, "object": "model"}]}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    input_ids = _prompt_ids(request.messages)
    gen_kwargs = dict(
        max_new_tokens=request.resolved_max_tokens,
        do_sample=request.temperature > 0,
        temperature=max(request.temperature, 1e-5),
        top_p=request.top_p,
        past_key_values=_build_cache(),
    )

    if request.stream:
        return StreamingResponse(_stream_response(input_ids, gen_kwargs, request.model),
                                   media_type="text/event-stream")

    with torch.no_grad():
        out = state.model.generate(input_ids, **gen_kwargs)
    completion_text = state.tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion_text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": input_ids.shape[1],
            "completion_tokens": out.shape[1] - input_ids.shape[1],
            "total_tokens": out.shape[1],
        },
    }


def _stream_response(input_ids, gen_kwargs, model_name: str):
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(state.tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {**gen_kwargs, "streamer": streamer}
    thread = threading.Thread(target=lambda: state.model.generate(input_ids, **gen_kwargs))
    thread.start()

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    for token_text in streamer:
        chunk = {
            "id": completion_id, "object": "chat.completion.chunk", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
        "id": completion_id, "object": "chat.completion.chunk", "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
    thread.join()


if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--k-bits", type=int, default=3)
    p.add_argument("--v-bits", type=int, default=4)
    p.add_argument("--use-prod", action="store_true")
    p.add_argument("--no-turboquant", action="store_true", help="Serve baseline BF16 KV-cache instead.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    load_real_model(ServerConfig(
        model_name_or_path=args.model, k_bits=args.k_bits, v_bits=args.v_bits,
        use_prod=args.use_prod, use_turboquant=not args.no_turboquant,
        host=args.host, port=args.port,
    ))
    uvicorn.run(app, host=args.host, port=args.port)
