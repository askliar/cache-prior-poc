from __future__ import annotations

import importlib.metadata
import json
import math
import platform
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F

from cacheprior.cache import ReplayMetrics, combine_replay_metrics, replay_belady
from cacheprior.config import ExperimentConfig, dump_resolved_config
from cacheprior.data import HFTextDataset
from cacheprior.models.olmoe import (
    OlmoeAdapter,
    load_hf_model_and_tokenizer,
    model_input_device,
)
from cacheprior.routing import RoutingController
from cacheprior.trace import RouteTrace, write_trace


def _json_dump(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "packages": {
            name: _package_version(name) for name in ("transformers", "datasets", "numpy", "PyYAML")
        },
    }


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _run_directory(config: ExperimentConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy = config.routing.policy
    if policy == "cache_prior":
        policy = f"{policy}-lambda-{config.routing.lambda_value:g}"
    dataset = _slug(
        "-".join(value for value in (config.dataset.source, config.dataset.subset) if value)
    )
    name = config.run_name or f"{dataset}-{policy}"
    run_id = f"{timestamp}-{_slug(name)}-{uuid4().hex[:8]}"
    output = Path(config.trace.output_dir).expanduser().resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "traces").mkdir()
    return output


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _live_metrics(traces: list[RouteTrace]) -> ReplayMetrics:
    if not traces:
        return ReplayMetrics(0, 0, 0, (), ())
    layers = traces[0].layers
    layer_accesses = tuple(
        sum(int(trace.hit_mask[layer].size) for trace in traces) for layer in range(layers)
    )
    layer_hits = tuple(
        sum(int(trace.hit_mask[layer].sum()) for trace in traces) for layer in range(layers)
    )
    accesses = sum(layer_accesses)
    hits = sum(layer_hits)
    return ReplayMetrics(
        accesses,
        hits,
        accesses - hits,
        layer_accesses,
        layer_hits,
    )


def _cache_metrics_with_transfer(
    metrics: ReplayMetrics,
    *,
    scored_tokens: int,
    expert_parameter_counts: dict[int, int],
    storage_bits: int,
) -> dict[str, Any]:
    result = metrics.to_dict()
    per_layer_misses = result["per_layer_misses"]
    if len(per_layer_misses) != len(expert_parameter_counts):
        raise ValueError("cache metrics and expert parameter layers do not align")
    parameters_fetched = sum(
        int(per_layer_misses[index]) * expert_parameter_counts[layer_id]
        for index, layer_id in enumerate(sorted(expert_parameter_counts))
    )
    estimated_bytes = parameters_fetched * storage_bits / 8.0
    result.update(
        {
            "misses_per_scored_token": (metrics.misses / scored_tokens if scored_tokens else 0.0),
            "parameters_fetched": parameters_fetched,
            "parameters_fetched_per_scored_token": (
                parameters_fetched / scored_tokens if scored_tokens else 0.0
            ),
            "storage_bits": storage_bits,
            "estimated_bytes_fetched": estimated_bytes,
            "estimated_bytes_per_scored_token": (
                estimated_bytes / scored_tokens if scored_tokens else 0.0
            ),
        }
    )
    return result


def _routing_metrics(traces: list[RouteTrace], top_j: int) -> dict[str, Any]:
    if not traces:
        return {
            "token_layer_events": 0,
            "changed_token_layer_events": 0,
            "changed_fraction": 0.0,
            "mean_set_divergence": 0.0,
            "top_j_retention": 1.0,
        }
    event_count = sum(trace.layers * trace.tokens for trace in traces)
    changed = sum(trace.changed_tokens for trace in traces)
    weighted_divergence = sum(
        trace.route_divergence * trace.layers * trace.tokens for trace in traces
    )
    retained = 0
    retention_total = 0
    if top_j:
        for trace in traces:
            protected = trace.original_ids[..., :top_j]
            for rank in range(top_j):
                retained += int(
                    np.any(
                        trace.selected_ids == protected[..., rank, None],
                        axis=-1,
                    ).sum()
                )
                retention_total += trace.layers * trace.tokens
    return {
        "token_layer_events": event_count,
        "changed_token_layer_events": changed,
        "changed_fraction": changed / event_count if event_count else 0.0,
        "mean_set_divergence": (weighted_divergence / event_count if event_count else 0.0),
        "top_j_retention": retained / retention_total if retention_total else 1.0,
    }


def run_experiment(
    config: ExperimentConfig,
    *,
    model: torch.nn.Module | None = None,
    tokenizer: Any | None = None,
) -> Path:
    """Run one dataset/routing configuration and return its run directory."""

    config.validate_static()
    _seed_everything(config.seed)
    owns_model = model is None
    if model is None or tokenizer is None:
        if model is not None or tokenizer is not None:
            raise ValueError("model and tokenizer must be supplied together")
        model, tokenizer = load_hf_model_and_tokenizer(config.model)
    model.eval()

    if config.model.adapter != "olmoe":
        raise ValueError("the initial implementation supports only adapter=olmoe")
    adapter = OlmoeAdapter(model)
    config.validate_for_model(top_k=adapter.top_k, num_experts=adapter.num_experts)
    controller = RoutingController(adapter.layer_specs, config.routing, config.cache)
    dataset = HFTextDataset(config.dataset, tokenizer)
    run_dir = _run_directory(config)

    dump_resolved_config(config, run_dir / "resolved_config.yaml")
    _json_dump(environment_manifest(), run_dir / "environment.json")
    dataset_manifest = dataset.manifest().to_dict()
    _json_dump(dataset_manifest, run_dir / "dataset_manifest.json")
    model_manifest = adapter.manifest()
    model_manifest.update(
        {
            "model_id": config.model.id,
            "revision": config.model.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "tokenizer_id": config.model.tokenizer_id or config.model.id,
            "tokenizer_resolved_revision": (getattr(tokenizer, "init_kwargs", {}) or {}).get(
                "_commit_hash"
            ),
            "dtype": config.model.dtype,
            "quantization": config.model.quantization,
            "device": config.model.device,
        }
    )
    _json_dump(model_manifest, run_dir / "model_manifest.json")

    total_nll = 0.0
    total_tokens = 0
    traces: list[RouteTrace] = []
    input_device = model_input_device(model)
    samples_path = run_dir / "per_window.jsonl"

    adapter.install(controller)
    try:
        with samples_path.open("w", encoding="utf-8") as sample_log:
            for window_index, window in enumerate(dataset):
                input_ids = window.input_ids.to(input_device)
                targets = window.target_ids.to(input_device)
                controller.begin_sequence(window.sample_id, window.scored_tokens)
                try:
                    with torch.inference_mode():
                        output = model(input_ids=input_ids, use_cache=False)
                        logits = output.logits
                    if tuple(logits.shape[:2]) != tuple(targets.shape):
                        raise RuntimeError(
                            f"LM logits shape {tuple(logits.shape)} does not align "
                            f"with targets {tuple(targets.shape)}"
                        )
                    nll = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                    trace = controller.end_sequence()
                except Exception:
                    controller.abort_sequence()
                    raise

                trace_path = run_dir / "traces" / f"{window_index:06d}.npz"
                write_trace(trace, trace_path)
                traces.append(trace)
                nll_value = float(nll.item())
                total_nll += nll_value
                total_tokens += window.scored_tokens
                record = {
                    "sample_id": window.sample_id,
                    "dataset_id": window.dataset_id,
                    "token_hash": window.token_hash,
                    "scored_tokens": window.scored_tokens,
                    "negative_log_likelihood": nll_value,
                    "perplexity": math.exp(nll_value / window.scored_tokens),
                    "cache_hits": trace.hits,
                    "cache_misses": trace.misses,
                    "route_divergence": trace.route_divergence,
                    "trace": str(trace_path.relative_to(run_dir)),
                }
                sample_log.write(json.dumps(record, sort_keys=True) + "\n")
    finally:
        adapter.restore()
        if owns_model and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not traces:
        raise RuntimeError("dataset produced no complete token windows")

    expert_counts = adapter.expert_parameter_counts()
    live = _live_metrics(traces)
    cache_results: dict[str, Any] = {
        "none": {
            "applicable": False,
            "note": "Quality-only baseline; no expert cache is simulated.",
        },
        "lru": _cache_metrics_with_transfer(
            live,
            scored_tokens=total_tokens,
            expert_parameter_counts=expert_counts,
            storage_bits=config.cache.storage_bits,
        ),
    }

    if config.routing.policy == "original":
        belady = combine_replay_metrics(
            replay_belady(
                trace.original_ids,
                trace.selected_weights,
                config.cache.capacity,
            )
            for trace in traces
        )
        cache_results["belady"] = _cache_metrics_with_transfer(
            belady,
            scored_tokens=total_tokens,
            expert_parameter_counts=expert_counts,
            storage_bits=config.cache.storage_bits,
        )

    metrics = {
        "schema_version": 1,
        "run_directory": str(run_dir),
        "dataset": dataset_manifest,
        "model": model_manifest,
        "routing": asdict(config.routing),
        "cache_config": asdict(config.cache),
        "quality": {
            "negative_log_likelihood": total_nll,
            "scored_tokens": total_tokens,
            "perplexity": math.exp(total_nll / total_tokens),
        },
        "cache": cache_results,
        "routing_change": _routing_metrics(traces, config.routing.top_j),
        "windows": len(traces),
    }
    _json_dump(metrics, run_dir / "metrics.json")
    return run_dir
