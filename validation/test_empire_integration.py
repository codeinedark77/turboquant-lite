"""
validation/test_empire_integration.py — tests what's actually testable in
empire_integration/local_llm_backend.py: health-check behavior and the
fallback logic. Does NOT test real Empire harness wiring, because the real
harness isn't available in this sandbox -- see that module's docstring.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from empire_integration.local_llm_backend import (
    LocalBackendConfig,
    is_local_backend_healthy,
    is_local_backend_healthy_async,
    get_llm_with_fallback,
)


def test_health_check_false_for_unreachable_server():
    cfg = LocalBackendConfig(base_url="http://127.0.0.1:1/v1", health_check_timeout_s=0.5)
    assert is_local_backend_healthy(cfg) is False


def test_fallback_uses_groq_when_local_unreachable():
    cfg = LocalBackendConfig(base_url="http://127.0.0.1:1/v1", health_check_timeout_s=0.5)
    sentinel_groq = object()
    result = get_llm_with_fallback(local_cfg=cfg, groq_llm=sentinel_groq)
    assert result is sentinel_groq


def test_fallback_raises_with_no_groq_and_no_local():
    cfg = LocalBackendConfig(base_url="http://127.0.0.1:1/v1", health_check_timeout_s=0.5)
    try:
        get_llm_with_fallback(local_cfg=cfg, groq_llm=None)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "unavailable" in str(e) or "not reachable" in str(e)


# --- Happy-path tests. Every test above this line only exercised the
# UNHEALTHY path -- a real gap, found by re-reading this file rather than
# assuming it was complete. Closing it needs a real server: the sync
# functions hit a real socket directly (httpx.get), which ASGITransport
# can't stand in for (it's async-only -- found earlier this session
# building the LangChain integration test). The async variants below CAN
# be tested this way, reliably; the sync path was separately spot-checked
# against an actual live server in this sandbox (10-27s to come up, once
# didn't respond inside 40s -- real timing variance, not something to
# build an automated test on).

def test_health_check_async_true_for_healthy_server():
    import asyncio
    import httpx
    from inference.quality_probe import build_synthetic_model
    from server.config import ServerConfig
    import server.app as server_app

    model, cfg_model = build_synthetic_model(seed=0)
    server_app.state.model = model
    server_app.state.model_config = cfg_model
    server_app.state.cfg = ServerConfig(model_name_or_path="test", device="cpu")

    async def run():
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=server_app.app), base_url="http://testserver")
        cfg = LocalBackendConfig(base_url="http://testserver/v1")
        return await is_local_backend_healthy_async(cfg, client=client)

    assert asyncio.run(run()) is True


def test_health_check_async_false_for_unreachable_server():
    import asyncio

    async def run():
        cfg = LocalBackendConfig(base_url="http://127.0.0.1:1/v1", health_check_timeout_s=0.5)
        return await is_local_backend_healthy_async(cfg)

    assert asyncio.run(run()) is False
