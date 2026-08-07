from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from cacheprior.config import ModelConfig
from cacheprior.routing import LayerRoutingSpec, RoutingController


@dataclass
class RouterHandle:
    layer_id: int
    parent: nn.Module
    attribute: str
    original: nn.Module
    spec: LayerRoutingSpec


class InstrumentedOlmoeRouter(nn.Module):
    """Preserve the installed OLMoE router contract while replacing top-k IDs."""

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

        for attribute in ("top_k", "num_experts", "norm_topk_prob", "hidden_dim"):
            if not hasattr(original, attribute):
                raise TypeError(f"OLMoE router is missing required attribute {attribute!r}")
            setattr(self, attribute, getattr(original, attribute))

    @property
    def weight(self) -> nn.Parameter:
        return self.original.weight  # type: ignore[no-any-return]

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Call the installed implementation so its first return value and
        # lambda=0 outputs remain version-faithful.
        first_output, original_scores, original_ids = self.original(hidden_states)
        flattened = hidden_states.reshape(-1, self.hidden_dim)
        if (
            isinstance(first_output, torch.Tensor)
            and first_output.numel() == flattened.shape[0] * self.num_experts
        ):
            # Reuse the exact router logits that produced original_ids. Repeating
            # the low-precision GEMM can perturb extremely close expert ties.
            raw_logits = first_output.reshape(-1, self.num_experts)
        else:
            raw_logits = F.linear(flattened, self.original.weight)
        original_probs = torch.softmax(raw_logits, dtype=torch.float32, dim=-1)
        selected_scores, selected_ids = self.controller.route(
            layer_id=self.layer_id,
            raw_logits=raw_logits,
            original_probs=original_probs,
            original_ids=original_ids,
            original_scores=original_scores,
        )
        return first_output, selected_scores, selected_ids


class OlmoeAdapter:
    model_type = "olmoe"

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles: list[RouterHandle] = self._discover()
        self._installed = False

    def _decoder_layers(self) -> Any:
        base = getattr(self.model, "model", None)
        layers = getattr(base, "layers", None)
        if layers is None:
            raise TypeError("expected an OLMoE causal LM with model.layers")
        return layers

    def _discover(self) -> list[RouterHandle]:
        handles: list[RouterHandle] = []
        for layer_id, layer in enumerate(self._decoder_layers()):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None)
            if gate is None:
                continue
            required = ("top_k", "num_experts", "norm_topk_prob", "weight")
            if not all(hasattr(gate, attribute) for attribute in required):
                raise TypeError(
                    f"layer {layer_id} gate {type(gate).__name__} is not a "
                    "supported OLMoE top-k router"
                )
            spec = LayerRoutingSpec(
                layer_id=layer_id,
                num_experts=int(gate.num_experts),
                top_k=int(gate.top_k),
                norm_topk_prob=bool(gate.norm_topk_prob),
            )
            handles.append(RouterHandle(layer_id, mlp, "gate", gate, spec))
        if not handles:
            raise TypeError("no OLMoE routers were discovered")

        reference = handles[0].spec
        for handle in handles[1:]:
            if (
                handle.spec.num_experts != reference.num_experts
                or handle.spec.top_k != reference.top_k
            ):
                raise ValueError("heterogeneous OLMoE routers are not supported yet")
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
                InstrumentedOlmoeRouter(
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
            layer = self._decoder_layers()[handle.layer_id]
            experts = layer.mlp.experts
            count = 0
            tensor_names = ("gate_up_proj", "down_proj")
            for name in tensor_names:
                tensor = getattr(experts, name, None)
                if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
                    if tensor.shape[0] != handle.spec.num_experts:
                        raise ValueError(
                            f"layer {handle.layer_id} {name} leading dimension "
                            "does not match expert count"
                        )
                    count += tensor[0].numel()
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


def _torch_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype {name!r}") from exc


def load_hf_model_and_tokenizer(config: ModelConfig) -> tuple[nn.Module, Any]:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_id = config.tokenizer_id or config.id
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        revision=config.tokenizer_revision or config.revision,
        trust_remote_code=config.trust_remote_code,
    )

    model_kwargs: dict[str, Any] = {
        "revision": config.revision,
        "trust_remote_code": config.trust_remote_code,
    }
    dtype_argument = (
        "dtype"
        if int(transformers.__version__.split(".", 1)[0]) >= 5
        else "torch_dtype"
    )
    model_kwargs[dtype_argument] = _torch_dtype(config.dtype)
    if config.attention_implementation:
        model_kwargs["attn_implementation"] = config.attention_implementation

    if config.quantization in {"8bit", "4bit"}:
        from transformers import BitsAndBytesConfig

        if config.quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif config.quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["device_map"] = "auto"
    elif config.quantization in {"native", "modelopt_fp8"}:
        # Preserve a checkpoint's own packed representation. This follows the
        # model publisher's from_pretrained(..., device_map="auto") contract
        # without applying BitsAndBytes or casting packed weights afterward.
        model_kwargs["device_map"] = "auto"

    if config.quantization == "modelopt_fp8":
        from transformers.utils import logging as transformers_logging

        from cacheprior.models.modelopt_fp8 import (
            apply_modelopt_fp8_checkpoint,
            validate_modelopt_loading_info,
        )

        original_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        try:
            model, loading_info = AutoModelForCausalLM.from_pretrained(
                config.id,
                output_loading_info=True,
                **model_kwargs,
            )
        finally:
            transformers_logging.set_verbosity(original_verbosity)
        validate_modelopt_loading_info(loading_info)
        wrapped_linears = apply_modelopt_fp8_checkpoint(
            model,
            model_id=config.id,
            revision=config.revision,
            output_dtype=_torch_dtype(config.dtype),
        )
        model._cacheprior_modelopt_fp8_linears = wrapped_linears
    else:
        model = AutoModelForCausalLM.from_pretrained(config.id, **model_kwargs)
    if config.quantization == "none":
        model.to(torch.device(config.device))
    model.eval()

    if hasattr(model, "set_experts_implementation"):
        # The proof relies on the Python router and expert path being visible.
        model.set_experts_implementation("eager")
    return model, tokenizer


def model_input_device(model: nn.Module) -> torch.device:
    embedding = model.get_input_embeddings()  # type: ignore[attr-defined]
    return embedding.weight.device
