"""server/config.py — runtime configuration for the pluggable server."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ServerConfig:
    model_name_or_path: str = "meta-llama/Llama-3.2-3B-Instruct"
    use_turboquant: bool = True
    k_bits: int = 3
    v_bits: int = 4
    # Phase 2b's bias-corrected K path underperformed plain MSE-K in every
    # full-model test run this session (see ARCHITECTURE.md Phase 2b status)
    # -- off by default until that's resolved on real weights.
    use_prod: bool = False
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8000
