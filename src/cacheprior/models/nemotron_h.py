from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cacheprior.routing import LayerRoutingSpec, RoutingController


@dataclass
class NemotronHRouterHandle:
    layer_id: int
    parent: nn.Module
    attribute: str
    original: nn.Module
    spec: LayerRoutingSpec


class InstrumentedNemotronHTopkRouter(nn.Module):
    """Preserve Nemotron-H routing weights while replacing selected expert IDs."""

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
            "config",
            "top_k",
            "n_routed_experts",
            "routed_scaling_factor",
            "n_group",
            "topk_group",
            "norm_topk_prob",
            "weight",
            "e_score_correction_bias",
        )
        for attribute in required:
            if not hasattr(original, attribute):
                raise TypeError(f"Nemotron-H router is missing required attribute {attribute!r}")
        if int(original.n_group) != 1 or int(original.topk_group) != 1:
            raise ValueError(
                "grouped Nemotron-H routing is not supported; expected n_group=topk_group=1"
            )

    @property
    def weight(self) -> nn.Parameter:
        return self.original.weight  # type: ignore[no-any-return]

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        original_ids, original_scores = self.original(hidden_states)
        flattened = hidden_states.reshape(-1, int(self.original.config.hidden_size))
        router_logits = F.linear(
            flattened.to(torch.float32),
            self.original.weight.to(torch.float32),
        )
        sigmoid_scores = router_logits.sigmoid()

        # Nemotron-H chooses experts using sigmoid scores plus its learned
        # correction bias. Cache-Prior operates on those exact ranking values;
        # mixing weights still come from the uncorrected sigmoid scores.
        routing_values = sigmoid_scores + self.original.e_score_correction_bias.to(
            device=sigmoid_scores.device,
            dtype=sigmoid_scores.dtype,
        )

        # The released gate requests sorted=False. Rank the selected set so
        # top-J refers to the J strongest original routing choices. Expert
        # accumulation is permutation invariant.
        order = torch.argsort(original_scores, dim=-1, descending=True)
        ranked_ids = original_ids.gather(-1, order)
        ranked_scores = original_scores.gather(-1, order)
        selected_scores, selected_ids = self.controller.route(
            layer_id=self.layer_id,
            raw_logits=routing_values,
            original_probs=sigmoid_scores,
            original_ids=ranked_ids,
            original_scores=ranked_scores,
        )
        return selected_ids, selected_scores


class NemotronHAdapter:
    """Adapter for NVIDIA Nemotron-H hybrid Mamba/attention MoE checkpoints."""

    model_type = "nemotron_h"

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles = self._discover()
        self._installed = False

    def _decoder_layers(self) -> Any:
        backbone = getattr(self.model, "backbone", None)
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise TypeError("expected a Nemotron-H causal LM with backbone.layers")
        return layers

    def _discover(self) -> list[NemotronHRouterHandle]:
        handles: list[NemotronHRouterHandle] = []
        for layer_id, layer in enumerate(self._decoder_layers()):
            mixer = getattr(layer, "mixer", None)
            gate = getattr(mixer, "gate", None)
            if gate is None:
                continue
            required = (
                "config",
                "top_k",
                "n_routed_experts",
                "norm_topk_prob",
                "routed_scaling_factor",
                "n_group",
                "topk_group",
                "weight",
                "e_score_correction_bias",
            )
            if not all(hasattr(gate, attribute) for attribute in required):
                raise TypeError(
                    f"layer {layer_id} gate {type(gate).__name__} is not a "
                    "supported Nemotron-H router"
                )
            if int(gate.n_group) != 1 or int(gate.topk_group) != 1:
                raise ValueError(
                    "grouped Nemotron-H routing is not supported; "
                    f"layer {layer_id} has n_group={gate.n_group}, "
                    f"topk_group={gate.topk_group}"
                )
            spec = LayerRoutingSpec(
                layer_id=layer_id,
                num_experts=int(gate.n_routed_experts),
                top_k=int(gate.top_k),
                norm_topk_prob=bool(gate.norm_topk_prob),
                score_scale=float(gate.routed_scaling_factor),
            )
            handles.append(NemotronHRouterHandle(layer_id, mixer, "gate", gate, spec))
        if not handles:
            raise TypeError("no Nemotron-H routers were discovered")

        reference = handles[0].spec
        for handle in handles[1:]:
            if (
                handle.spec.num_experts != reference.num_experts
                or handle.spec.top_k != reference.top_k
            ):
                raise ValueError("heterogeneous Nemotron-H routers are not supported")
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
                InstrumentedNemotronHTopkRouter(
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
            config = handle.original.config
            hidden_size = int(config.hidden_size)
            intermediate_size = int(config.moe_intermediate_size)
            count = 2 * hidden_size * intermediate_size
            if bool(getattr(config, "mlp_bias", False)):
                count += hidden_size + intermediate_size
            counts[handle.layer_id] = count
        return counts

    def manifest(self) -> dict[str, Any]:
        return {
            "adapter": self.model_type,
            "model_class": type(self.model).__name__,
            "modelopt_fp8_linears": getattr(
                self.model,
                "_cacheprior_modelopt_fp8_linears",
                None,
            ),
            "num_moe_layers": len(self._handles),
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "routing_value": "sigmoid_score_plus_correction_bias",
            "layers": [asdict(spec) for spec in self.layer_specs],
            "expert_parameters_per_layer": self.expert_parameter_counts(),
        }
