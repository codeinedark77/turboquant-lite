"""
empire_integration/local_llm_backend.py

Wires turboquant-lite's server into a LangGraph-based harness as an
alternate LLM backend alongside Groq.

HONEST SCOPE: this is designed from memory of the Empire's architecture
(LangGraph harness, A2A bus, Safety Guardian gate, unified capability
registry, 19 agents / 14 dispatchable + 5 privileged) -- the actual
harness/registry source isn't available in this sandbox, so this file is a
starting design to verify against the real code, not confirmed-working
integration code the way turboquant-lite/ itself is. Everything else in
this repo is tested end-to-end against something real (a synthetic model,
or in this file's case, a real client library); this file's only tested
claim is the one below.

What IS actually tested (validation/test_server.py::
test_real_chatopenai_client_respects_token_limit): a real
langchain_openai.ChatOpenAI client talking to a real running
turboquant-lite server works correctly end to end -- and caught a real bug
doing it (ChatOpenAI sends `max_completion_tokens`, not `max_tokens`; fixed
in server/app.py, now regression-tested). That's the actual interface
contract everything below builds on.

NOT a quality-equivalent swap for Groq: this serves whatever model you
point it at locally (Llama-3.2-3B-Instruct in this project's testing) vs.
Groq's llama-3.3-70b-versatile elsewhere in the stack. A ~3B model is not
a ~70B model. This is for scenarios where local/offline/free inference
matters more than peak reasoning quality -- a narrow-scope dispatchable
agent, a dev/offline mode, or a fallback when Groq is rate-limited or
down -- not a wholesale replacement for agents doing complex reasoning.
Which agents (if any) are appropriate candidates is a call to make against
your actual capability registry, not something this file decides.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LocalBackendConfig:
    base_url: str = "http://localhost:8000/v1"
    model_name: str = "turboquant-lite"
    health_check_timeout_s: float = 2.0
    max_tokens: int = 1024
    temperature: float = 0.7


async def is_local_backend_healthy_async(cfg: LocalBackendConfig, client=None) -> bool:
    """Async twin of is_local_backend_healthy. Exists specifically because
    the sync version, tested against a real live socket in this sandbox,
    turned out to take anywhere from ~10s to 27s+ to become reachable (once
    didn't come up inside 40s at all) -- real variance, not a bug, but not
    something to build an automated test on top of. This version is
    reliably testable via httpx.AsyncClient + ASGITransport instead, no
    real socket needed, matching how server/app.py's own request/response
    plumbing gets tested."""
    import httpx
    client = client or httpx.AsyncClient()
    try:
        resp = await client.get(cfg.base_url.rsplit("/v1", 1)[0] + "/v1/models",
                                  timeout=cfg.health_check_timeout_s)
        return resp.status_code == 200
    except httpx.HTTPError as e:
        logger.warning(f"turboquant-lite health check failed: {e}")
        return False


def is_local_backend_healthy(cfg: LocalBackendConfig, client=None) -> bool:
    """Actual liveness check, not an assumption -- the whole point of the
    additive/fallback design is that a dead local server degrades to Groq
    instead of taking an agent down. Cheap: hits /v1/models, not a real
    generation.

    `client` is injectable (any object with a sync `.get(url, timeout=...)`)
    so this is actually testable in-process via httpx's ASGI transport,
    rather than only against a real socket -- found this gap by trying to
    test the healthy path and discovering the function couldn't be pointed
    at anything but a live URL."""
    import httpx
    client = client or httpx
    try:
        resp = client.get(cfg.base_url.rsplit("/v1", 1)[0] + "/v1/models",
                           timeout=cfg.health_check_timeout_s)
        return resp.status_code == 200
    except httpx.HTTPError as e:
        logger.warning(f"turboquant-lite health check failed: {e}")
        return False


def get_local_llm(cfg: Optional[LocalBackendConfig] = None, health_client=None, http_async_client=None):
    """Returns a langchain_openai.ChatOpenAI pointed at the turboquant-lite
    server. Raises RuntimeError if the server isn't reachable -- callers
    that want silent fallback to Groq should use get_llm_with_fallback
    instead of catching this themselves in N different places (that
    per-call-site pattern is exactly how the == 19 hardcoded-count bug
    class spread across four files; one fallback path, used everywhere,
    stays a single thing to fix if it's ever wrong).

    health_client/http_async_client: injectable for testing, as above."""
    from langchain_openai import ChatOpenAI

    cfg = cfg or LocalBackendConfig()
    if not is_local_backend_healthy(cfg, client=health_client):
        raise RuntimeError(f"turboquant-lite server not reachable at {cfg.base_url}")

    kwargs = dict(
        base_url=cfg.base_url,
        api_key="not-needed",
        model=cfg.model_name,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client
    return ChatOpenAI(**kwargs)


def get_llm_with_fallback(local_cfg: Optional[LocalBackendConfig] = None, groq_llm=None,
                            health_client=None, http_async_client=None):
    """local_cfg's model if healthy, else groq_llm. groq_llm is passed in
    rather than constructed here -- this module shouldn't need to know your
    Groq client setup (API key handling, model choice) to do its one job.
    health_client/http_async_client: injectable for testing (see get_local_llm).

    Verify against your actual harness before wiring this into dispatch:
    - Where does agent -> LLM-backend selection actually live? If it's the
      capability registry, this backend choice belongs there as a real,
      checked field -- not a new fifth list alongside the three you already
      unified, and not a flag that's declared but never read by score_all()
      the way category="privileged" currently is.
    - Does Safety Guardian care about backend identity at all? From what I
      know of it, it gates by agent (github_agent, content_agent,
      browser_agent, computer_use) based on what an agent *does*, not which
      model reasons for it -- swapping backend shouldn't need a new gate,
      but that's an inference from memory of the gate's shape, not a read
      of its actual code, so confirm rather than assume.
    - A2A bus: if inter-agent messages carry any assumption about response
      latency or format tied to Groq specifically, a 3B local model's
      latency/output profile may not match -- worth a spot check, not
      assumed fine.
    """
    if local_cfg is not None and is_local_backend_healthy(local_cfg, client=health_client):
        try:
            return get_local_llm(local_cfg, health_client=health_client, http_async_client=http_async_client)
        except RuntimeError:
            pass
    if groq_llm is None:
        raise RuntimeError("local backend unavailable and no groq_llm fallback provided")
    return groq_llm
