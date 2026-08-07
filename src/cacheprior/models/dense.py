from __future__ import annotations

from typing import Any

from torch import nn


class DenseAdapter:
    """Metadata-only adapter for perplexity evaluation of dense causal LMs."""

    model_type = "dense"
    is_moe = False

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter": self.model_type,
            "model_class": type(self.model).__name__,
            "num_moe_layers": 0,
            "cache_policies_applicable": False,
        }
