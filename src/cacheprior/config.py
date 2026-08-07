from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str | None = None
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    adapter: str = "olmoe"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    attention_implementation: str | None = "sdpa"
    quantization: str = "none"
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    subset: str | None = None
    split: str = "validation"
    revision: str | None = None
    text_field: str = "text"
    mode: str = "concatenate"
    separator: str = "\n\n"
    join_before_tokenization: bool = False
    prediction_length: int = 512
    max_windows: int | None = 32
    streaming: bool = False


@dataclass(frozen=True)
class RoutingConfig:
    policy: str = "original"
    lambda_value: float = 0.0
    top_j: int = 2
    range_estimator: str = "inclusive_running_mean"


@dataclass(frozen=True)
class CacheConfig:
    policy: str = "lru"
    capacity: int = 32
    initial_state: str = "empty"
    reset: str = "per_window"
    update_order: str = "descending_original_probability"
    storage_bits: int = 16


@dataclass(frozen=True)
class TraceConfig:
    level: str = "compact"
    output_dir: str = "runs"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    dataset: DatasetConfig
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    seed: int = 1234
    run_name: str | None = None

    def validate_static(self) -> None:
        if self.routing.policy not in {"original", "cache_prior"}:
            raise ValueError("routing.policy must be 'original' or 'cache_prior'")
        if self.routing.lambda_value < 0.0:
            raise ValueError("routing.lambda must be non-negative")
        if self.routing.top_j < 0:
            raise ValueError("routing.top_j must be non-negative")
        if self.routing.range_estimator != "inclusive_running_mean":
            raise ValueError("only inclusive_running_mean is supported")
        if self.cache.policy != "lru":
            raise ValueError("live runs currently require cache.policy=lru")
        if self.cache.capacity <= 0:
            raise ValueError("cache.capacity must be positive")
        if self.cache.initial_state != "empty":
            raise ValueError("only an empty initial cache is supported")
        if self.cache.reset != "per_window":
            raise ValueError("only per_window cache reset is supported")
        if self.cache.update_order != "descending_original_probability":
            raise ValueError("only descending_original_probability update order is supported")
        if self.cache.storage_bits not in {4, 8, 16, 32}:
            raise ValueError("cache.storage_bits must be one of 4, 8, 16, or 32")
        if self.dataset.mode not in {"concatenate", "document"}:
            raise ValueError("dataset.mode must be 'concatenate' or 'document'")
        if self.dataset.join_before_tokenization and self.dataset.mode != "concatenate":
            raise ValueError("join_before_tokenization requires dataset.mode=concatenate")
        if self.dataset.join_before_tokenization and self.dataset.streaming:
            raise ValueError("join_before_tokenization is not supported for streaming datasets")
        if self.dataset.prediction_length <= 0:
            raise ValueError("dataset.prediction_length must be positive")
        if self.dataset.max_windows is not None and self.dataset.max_windows <= 0:
            raise ValueError("dataset.max_windows must be positive or null")
        if self.model.quantization not in {
            "none",
            "8bit",
            "4bit",
            "native",
            "modelopt_fp8",
        }:
            raise ValueError(
                "model.quantization must be none, 8bit, 4bit, native, or modelopt_fp8"
            )

    def validate_for_model(self, *, top_k: int, num_experts: int) -> None:
        self.validate_static()
        if self.cache.capacity < top_k:
            raise ValueError(
                f"cache capacity ({self.cache.capacity}) must be >= model top-k ({top_k})"
            )
        if self.cache.capacity > num_experts:
            raise ValueError(
                f"cache capacity ({self.cache.capacity}) exceeds expert count ({num_experts})"
            )
        if self.routing.top_j >= top_k:
            raise ValueError(
                f"routing.top_j ({self.routing.top_j}) must be smaller than top-k ({top_k})"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["routing"]["lambda"] = data["routing"].pop("lambda_value")
        return data

    def with_dataset(self, dataset: DatasetConfig) -> ExperimentConfig:
        return replace(self, dataset=dataset)

    def with_routing(
        self,
        *,
        policy: str,
        lambda_value: float | None = None,
        top_j: int | None = None,
    ) -> ExperimentConfig:
        routing = replace(
            self.routing,
            policy=policy,
            lambda_value=self.routing.lambda_value if lambda_value is None else lambda_value,
            top_j=self.routing.top_j if top_j is None else top_j,
        )
        return replace(self, routing=routing)


def _routing_from_dict(data: dict[str, Any]) -> RoutingConfig:
    normalized = dict(data)
    if "lambda" in normalized:
        normalized["lambda_value"] = normalized.pop("lambda")
    return RoutingConfig(**normalized)


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    config = ExperimentConfig(
        model=ModelConfig(**data["model"]),
        dataset=DatasetConfig(**data["dataset"]),
        routing=_routing_from_dict(data.get("routing", {})),
        cache=CacheConfig(**data.get("cache", {})),
        trace=TraceConfig(**data.get("trace", {})),
        seed=int(data.get("seed", 1234)),
        run_name=data.get("run_name"),
    )
    config.validate_static()
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return experiment_config_from_dict(data)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if "dataset" in data:
        data = data["dataset"]
    return DatasetConfig(**data)


def dump_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
