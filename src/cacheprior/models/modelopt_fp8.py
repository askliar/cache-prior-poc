from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


class ModelOptFp8Linear(nn.Module):
    """Static per-tensor W8A8 linear from a ModelOpt unified checkpoint."""

    def __init__(
        self,
        original: nn.Linear,
        *,
        input_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if original.weight.dtype not in _FP8_DTYPES:
            raise TypeError(f"expected an FP8 weight, got {original.weight.dtype}")
        if input_scale.numel() != 1 or weight_scale.numel() != 1:
            raise ValueError("only static per-tensor ModelOpt FP8 scales are supported")

        self.in_features = original.in_features
        self.out_features = original.out_features
        self.output_dtype = output_dtype
        self.weight = original.weight
        self.bias = original.bias
        self.register_buffer(
            "input_scale",
            input_scale.reshape(1).to(device=original.weight.device, dtype=torch.float32),
        )
        self.register_buffer(
            "weight_scale",
            weight_scale.reshape(1).to(device=original.weight.device, dtype=torch.float32),
        )

    def _quantize_input(self, value: torch.Tensor) -> torch.Tensor:
        fp8_limit = torch.finfo(self.weight.dtype).max
        return (
            value.to(torch.float32)
            .mul(self.input_scale.reciprocal())
            .clamp(-fp8_limit, fp8_limit)
            .to(self.weight.dtype)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_shape = value.shape
        flattened = value.reshape(-1, original_shape[-1])
        quantized = self._quantize_input(flattened)
        output = torch._scaled_mm(
            quantized,
            self.weight.t(),
            scale_a=self.input_scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=self.output_dtype,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.reshape(*original_shape[:-1], self.out_features)


def _snapshot_path(model_id: str, revision: str | None) -> Path:
    local = Path(model_id).expanduser()
    if local.is_dir():
        return local.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=("model*.safetensors", "model.safetensors.index.json"),
        )
    )


def load_modelopt_fp8_scales(
    model_id: str,
    revision: str | None,
    required_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Load only requested scale tensors from a sharded unified checkpoint."""

    snapshot = _snapshot_path(model_id, revision)
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"ModelOpt FP8 checkpoint index not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map")

    by_shard: dict[str, list[str]] = defaultdict(list)
    missing_from_index = []
    for key in sorted(required_keys):
        shard = weight_map.get(key)
        if shard is None:
            missing_from_index.append(key)
        else:
            by_shard[str(shard)].append(key)
    if missing_from_index:
        raise KeyError(f"FP8 scale tensors are absent from the index: {missing_from_index[:4]}")

    from safetensors import safe_open

    scales: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                scales[key] = handle.get_tensor(key)
    return scales


def wrap_modelopt_fp8_linears(
    model: nn.Module,
    scales: dict[str, torch.Tensor],
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Replace every FP8 nn.Linear while preserving module names and parameters."""

    linears = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.dtype in _FP8_DTYPES
    ]
    if not linears:
        raise TypeError("the checkpoint did not instantiate any FP8 nn.Linear modules")

    for name, module in linears:
        input_key = f"{name}.input_scale"
        weight_key = f"{name}.weight_scale"
        try:
            input_scale = scales[input_key]
            weight_scale = scales[weight_key]
        except KeyError as exc:
            raise KeyError(f"missing ModelOpt scale for FP8 linear {name!r}") from exc

        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attribute,
            ModelOptFp8Linear(
                module,
                input_scale=input_scale,
                weight_scale=weight_scale,
                output_dtype=output_dtype,
            ),
        )
    return len(linears)


def apply_modelopt_fp8_checkpoint(
    model: nn.Module,
    *,
    model_id: str,
    revision: str | None,
    output_dtype: torch.dtype,
) -> int:
    required_keys: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.dtype in _FP8_DTYPES:
            required_keys.add(f"{name}.input_scale")
            required_keys.add(f"{name}.weight_scale")
    scales = load_modelopt_fp8_scales(model_id, revision, required_keys)
    return wrap_modelopt_fp8_linears(model, scales, output_dtype=output_dtype)


def validate_modelopt_loading_info(loading_info: dict[str, Any]) -> None:
    """Reject missing model weights or ignored tensors other than FP8 scales."""

    missing = list(loading_info.get("missing_keys", ()))
    mismatched = list(loading_info.get("mismatched_keys", ()))
    unexpected = list(loading_info.get("unexpected_keys", ()))
    non_scale = [
        key
        for key in unexpected
        if not key.endswith((".input_scale", ".weight_scale"))
    ]
    if missing or mismatched or non_scale:
        raise RuntimeError(
            "invalid ModelOpt FP8 checkpoint load: "
            f"missing={missing[:4]}, mismatched={mismatched[:4]}, "
            f"unexpected_non_scale={non_scale[:4]}"
        )
