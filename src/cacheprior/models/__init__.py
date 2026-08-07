from __future__ import annotations

from torch import nn

from cacheprior.models.deepseek_v2 import DeepseekV2Adapter
from cacheprior.models.dense import DenseAdapter
from cacheprior.models.nemotron_h import NemotronHAdapter
from cacheprior.models.olmoe import OlmoeAdapter, load_hf_model_and_tokenizer
from cacheprior.models.qwen2_moe import Qwen2MoeAdapter


def create_model_adapter(
    model: nn.Module,
    adapter: str,
) -> DenseAdapter | OlmoeAdapter | Qwen2MoeAdapter | DeepseekV2Adapter | NemotronHAdapter:
    adapters = {
        "dense": DenseAdapter,
        "olmoe": OlmoeAdapter,
        "qwen2_moe": Qwen2MoeAdapter,
        "deepseek_v2": DeepseekV2Adapter,
        "nemotron_h": NemotronHAdapter,
    }
    try:
        adapter_type = adapters[adapter]
    except KeyError as exc:
        raise ValueError(f"unsupported model adapter {adapter!r}") from exc
    return adapter_type(model)


__all__ = [
    "DenseAdapter",
    "DeepseekV2Adapter",
    "NemotronHAdapter",
    "OlmoeAdapter",
    "Qwen2MoeAdapter",
    "create_model_adapter",
    "load_hf_model_and_tokenizer",
]
