from __future__ import annotations

from cacheprior.models.olmoe import OlmoeAdapter


class Qwen2MoeAdapter(OlmoeAdapter):
    """Adapter for Qwen1.5/Qwen2 MoE's Transformers top-k router."""

    model_type = "qwen2_moe"
