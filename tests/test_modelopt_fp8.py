from __future__ import annotations

import pytest
import torch
from torch import nn

from cacheprior.models.modelopt_fp8 import (
    ModelOptFp8Linear,
    validate_modelopt_loading_info,
    wrap_modelopt_fp8_linears,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False, dtype=torch.float32)
        self.proj.weight = nn.Parameter(
            self.proj.weight.detach().to(torch.float8_e4m3fn),
            requires_grad=False,
        )


def test_wrap_modelopt_fp8_linears_preserves_weight_and_scales() -> None:
    model = TinyModel()
    original_weight = model.proj.weight
    count = wrap_modelopt_fp8_linears(
        model,
        {
            "proj.input_scale": torch.tensor([0.25]),
            "proj.weight_scale": torch.tensor([0.5]),
        },
    )

    assert count == 1
    assert isinstance(model.proj, ModelOptFp8Linear)
    assert model.proj.weight is original_weight
    assert model.proj.output_dtype == torch.bfloat16
    assert model.proj.input_scale.item() == pytest.approx(0.25)
    assert model.proj.weight_scale.item() == pytest.approx(0.5)


def test_wrap_modelopt_fp8_linears_requires_both_scales() -> None:
    with pytest.raises(KeyError, match="missing ModelOpt scale"):
        wrap_modelopt_fp8_linears(
            TinyModel(),
            {"proj.input_scale": torch.tensor([0.25])},
        )


def test_validate_modelopt_loading_info_allows_only_scale_metadata() -> None:
    validate_modelopt_loading_info(
        {
            "missing_keys": [],
            "mismatched_keys": [],
            "unexpected_keys": ["proj.input_scale", "proj.weight_scale"],
        }
    )

    with pytest.raises(RuntimeError, match="unexpected_non_scale"):
        validate_modelopt_loading_info(
            {
                "missing_keys": [],
                "mismatched_keys": [],
                "unexpected_keys": ["proj.weight"],
            }
        )
