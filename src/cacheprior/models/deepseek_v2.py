from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cacheprior.routing import LayerRoutingSpec, RoutingController


@dataclass
class DeepseekRouterHandle:
    layer_id: int
    parent: nn.Module
    attribute: str
    original: nn.Module
    spec: LayerRoutingSpec


class InstrumentedDeepseekV2Gate(nn.Module):
    """Preserve DeepSeek-V2's gate contract while replacing routed expert IDs."""

    def __init__(
        self,
        original: nn.Module,
        layer_id: int,
        controller: RoutingController,
    ) -> None:
        super().__init__()
        self.original = original
        self.layer_id = layer_id
        self.controller = controller

        required = (
            "top_k",
            "n_routed_experts",
            "norm_topk_prob",
            "gating_dim",
            "routed_scaling_factor",
            "weight",
        )
        for attribute in required:
            if not hasattr(original, attribute):
                raise TypeError(f"DeepSeek-V2 gate is missing required attribute {attribute!r}")

    @property
    def weight(self) -> nn.Parameter:
        return self.original.weight  # type: ignore[no-any-return]

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        original_ids, original_scores, aux_loss = self.original(hidden_states)
        flattened = hidden_states.reshape(-1, self.original.gating_dim)
        raw_logits = F.linear(
            flattened.to(torch.float32),
            self.original.weight.to(torch.float32),
        )
        original_probs = torch.softmax(raw_logits, dtype=torch.float32, dim=-1)

        # DeepSeek uses torch.topk(..., sorted=False). Sorting the selected set
        # makes "top-J" mean the J highest-probability experts. The MoE weighted
        # sum is permutation invariant, so the original policy remains faithful.
        order = torch.argsort(original_scores, dim=-1, descending=True)
        ranked_ids = original_ids.gather(-1, order)
        ranked_scores = original_scores.gather(-1, order)
        selected_scores, selected_ids = self.controller.route(
            layer_id=self.layer_id,
            raw_logits=raw_logits,
            original_probs=original_probs,
            original_ids=ranked_ids,
            original_scores=ranked_scores,
        )
        return selected_ids, selected_scores, aux_loss


class DeepseekV2Adapter:
    model_type = "deepseek_v2"

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles = self._discover()
        self._installed = False

    def _decoder_layers(self) -> Any:
        base = getattr(self.model, "model", None)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise TypeError("expected a DeepSeek-V2 causal LM with model.layers")
        return layers

    def _discover(self) -> list[DeepseekRouterHandle]:
        handles: list[DeepseekRouterHandle] = []
        for layer_id, layer in enumerate(self._decoder_layers()):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None)
            if gate is None:
                continue
            required = (
                "top_k",
                "n_routed_experts",
                "norm_topk_prob",
                "routed_scaling_factor",
                "weight",
            )
            if not all(hasattr(gate, attribute) for attribute in required):
                raise TypeError(
                    f"layer {layer_id} gate {type(gate).__name__} is not a "
                    "supported DeepSeek-V2 router"
                )
            spec = LayerRoutingSpec(
                layer_id=layer_id,
                num_experts=int(gate.n_routed_experts),
                top_k=int(gate.top_k),
                norm_topk_prob=bool(gate.norm_topk_prob),
                score_scale=float(gate.routed_scaling_factor),
            )
            handles.append(DeepseekRouterHandle(layer_id, mlp, "gate", gate, spec))
        if not handles:
            raise TypeError("no DeepSeek-V2 routers were discovered")

        reference = handles[0].spec
        for handle in handles[1:]:
            if (
                handle.spec.num_experts != reference.num_experts
                or handle.spec.top_k != reference.top_k
            ):
                raise ValueError("heterogeneous DeepSeek-V2 routers are not supported")
        return handles

    @property
    def layer_specs(self) -> tuple[LayerRoutingSpec, ...]:
        return tuple(handle.spec for handle in self._handles)

    @property
    def num_experts(self) -> int:
        return self.layer_specs[0].num_experts

    @property
    def top_k(self) -> int:
        return self.layer_specs[0].top_k

    def install(self, controller: RoutingController) -> None:
        if self._installed:
            raise RuntimeError("router wrappers are already installed")
        for handle in self._handles:
            setattr(
                handle.parent,
                handle.attribute,
                InstrumentedDeepseekV2Gate(
                    handle.original,
                    handle.layer_id,
                    controller,
                ),
            )
        self._installed = True

    def restore(self) -> None:
        if not self._installed:
            return
        for handle in self._handles:
            setattr(handle.parent, handle.attribute, handle.original)
        self._installed = False

    def expert_parameter_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for handle in self._handles:
            experts = self._decoder_layers()[handle.layer_id].mlp.experts
            expert = next((candidate for candidate in experts if candidate is not None), None)
            if expert is None:
                raise TypeError(f"layer {handle.layer_id} contains no local routed experts")
            count = sum(parameter.numel() for parameter in expert.parameters())
            if count == 0:
                raise TypeError(
                    f"cannot derive per-expert parameter count for layer {handle.layer_id}"
                )
            counts[handle.layer_id] = int(count)
        return counts

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter": self.model_type,
            "model_class": type(self.model).__name__,
            "num_moe_layers": len(self._handles),
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "layers": [asdict(spec) for spec in self.layer_specs],
            "expert_parameters_per_layer": self.expert_parameter_counts(),
        }
