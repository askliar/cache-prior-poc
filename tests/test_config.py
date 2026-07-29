from pathlib import Path

import pytest

from cacheprior.config import load_experiment_config


def test_example_config_loads() -> None:
    root = Path(__file__).parents[1]
    config = load_experiment_config(root / "configs" / "experiments" / "smoke.yaml")
    assert config.model.adapter == "olmoe"
    assert config.routing.lambda_value == 0.5
    assert config.dataset.prediction_length == 64


def test_model_constraints_are_validated() -> None:
    root = Path(__file__).parents[1]
    config = load_experiment_config(root / "configs" / "experiments" / "smoke.yaml")
    with pytest.raises(ValueError, match="capacity"):
        config.validate_for_model(top_k=40, num_experts=64)
