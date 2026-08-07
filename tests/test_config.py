from dataclasses import replace
from pathlib import Path

import pytest

from cacheprior.config import load_experiment_config


def test_example_config_loads() -> None:
    root = Path(__file__).parents[1]
    config = load_experiment_config(root / "configs" / "experiments" / "smoke.yaml")
    assert config.model.adapter == "olmoe"
    assert config.routing.lambda_value == 0.5
    assert config.dataset.prediction_length == 64


def test_paper_configs_match_reported_half_cache_protocol() -> None:
    root = Path(__file__).parents[1]
    qwen = load_experiment_config(
        root / "configs" / "experiments" / "paper-qwen-wikitext.yaml"
    )
    deepseek = load_experiment_config(
        root / "configs" / "experiments" / "paper-deepseek-wikitext.yaml"
    )
    assert qwen.model.adapter == "qwen2_moe"
    assert qwen.cache.capacity == 30
    assert deepseek.model.adapter == "deepseek_v2"
    assert deepseek.cache.capacity == 32
    for config in (qwen, deepseek):
        assert config.dataset.join_before_tokenization
        assert config.dataset.prediction_length == 1024
        assert config.dataset.max_windows is None
        assert config.routing.top_j == 2
        assert config.cache.update_order == "descending_original_probability"


def test_nemotron_config_matches_released_checkpoint() -> None:
    root = Path(__file__).parents[1]
    config = load_experiment_config(
        root / "configs" / "experiments" / "nemotron3-nano-nvfp4.yaml"
    )

    assert config.model.id == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
    assert config.model.revision == "ce1b118ae66ec705d02c241525192832eb045fd3"
    assert config.model.adapter == "nemotron_h"
    assert config.model.quantization == "native"
    assert config.model.trust_remote_code
    assert config.cache.capacity == 64
    assert config.cache.storage_bits == 4
    config.validate_for_model(top_k=6, num_experts=128)


def test_fp8_configs_are_pinned_and_use_expected_loading() -> None:
    root = Path(__file__).parents[1]
    qwen = load_experiment_config(
        root / "configs" / "experiments" / "qwen3-8b-fp8-wikitext.yaml"
    )
    nemotron = load_experiment_config(
        root / "configs" / "experiments" / "nemotron3-nano-fp8.yaml"
    )

    assert qwen.model.id == "Qwen/Qwen3-8B-FP8"
    assert qwen.model.adapter == "dense"
    assert qwen.model.quantization == "native"
    assert qwen.routing.policy == "original"
    assert qwen.dataset.max_windows is None

    assert nemotron.model.id == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
    assert nemotron.model.adapter == "nemotron_h"
    assert nemotron.model.quantization == "modelopt_fp8"
    assert nemotron.cache.capacity == 64
    assert nemotron.cache.storage_bits == 8
    nemotron.validate_for_model(top_k=6, num_experts=128)


def test_model_constraints_are_validated() -> None:
    root = Path(__file__).parents[1]
    config = load_experiment_config(root / "configs" / "experiments" / "smoke.yaml")
    with pytest.raises(ValueError, match="capacity"):
        config.validate_for_model(top_k=40, num_experts=64)


def test_lambda_can_be_extrapolated_above_one() -> None:
    root = Path(__file__).parents[1]
    config = load_experiment_config(root / "configs" / "experiments" / "smoke.yaml")
    extrapolated = replace(
        config,
        routing=replace(config.routing, lambda_value=3.0),
    )
    extrapolated.validate_static()
