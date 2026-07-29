from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from cacheprior.cache import LRUCache
from cacheprior.config import CacheConfig, RoutingConfig
from cacheprior.trace import RouteTrace

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class LayerRoutingSpec:
    layer_id: int
    num_experts: int
    top_k: int
    norm_topk_prob: bool


class RunningMean:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0

    def update(self, value: float) -> float:
        self.count += 1
        self.total += float(value)
        return self.total / self.count

    @property
    def value(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass
class _LayerBuffer:
    original_ids: list[np.ndarray]
    selected_ids: list[np.ndarray]
    selected_weights: list[np.ndarray]
    hit_mask: list[np.ndarray]
    logit_range: list[float]
    range_mean: list[float]

    @classmethod
    def empty(cls) -> _LayerBuffer:
        return cls([], [], [], [], [], [])


class RoutingController:
    """Owns per-layer cache/range state and applies routing token by token."""

    def __init__(
        self,
        layer_specs: Sequence[LayerRoutingSpec],
        routing: RoutingConfig,
        cache: CacheConfig,
    ) -> None:
        if not layer_specs:
            raise ValueError("at least one MoE layer is required")
        self.layer_specs = tuple(sorted(layer_specs, key=lambda spec: spec.layer_id))
        self.routing = routing
        self.cache_config = cache
        self._spec_by_id = {spec.layer_id: spec for spec in self.layer_specs}
        if len(self._spec_by_id) != len(self.layer_specs):
            raise ValueError("layer IDs must be unique")
        self._caches = {spec.layer_id: LRUCache(cache.capacity) for spec in self.layer_specs}
        self._range_means = {spec.layer_id: RunningMean() for spec in self.layer_specs}
        self._active_sample: str | None = None
        self._expected_tokens = 0
        self._buffers: dict[int, _LayerBuffer] = {}
        self._token_cursors: dict[int, int] = {}

    def begin_sequence(self, sample_id: str, expected_tokens: int) -> None:
        if self._active_sample is not None:
            raise RuntimeError(f"sequence {self._active_sample!r} is still active")
        if expected_tokens <= 0:
            raise ValueError("expected_tokens must be positive")
        self._active_sample = sample_id
        self._expected_tokens = expected_tokens
        self._buffers = {spec.layer_id: _LayerBuffer.empty() for spec in self.layer_specs}
        self._token_cursors = {spec.layer_id: 0 for spec in self.layer_specs}
        for cache in self._caches.values():
            cache.reset()

    def _assert_router_inputs(
        self,
        layer_id: int,
        raw_logits: torch.Tensor,
        original_probs: torch.Tensor,
        original_ids: torch.Tensor,
        original_scores: torch.Tensor,
    ) -> LayerRoutingSpec:
        if self._active_sample is None:
            raise RuntimeError("router called outside an active sequence")
        try:
            spec = self._spec_by_id[layer_id]
        except KeyError as exc:
            raise ValueError(f"unknown layer ID {layer_id}") from exc
        if raw_logits.ndim != 2 or raw_logits.shape[1] != spec.num_experts:
            raise ValueError(
                f"layer {layer_id} raw logits must have shape [tokens, {spec.num_experts}]"
            )
        if original_probs.shape != raw_logits.shape:
            raise ValueError("original probability shape mismatch")
        expected_topk_shape = (raw_logits.shape[0], spec.top_k)
        if tuple(original_ids.shape) != expected_topk_shape:
            raise ValueError("original expert ID shape mismatch")
        if tuple(original_scores.shape) != expected_topk_shape:
            raise ValueError("original score shape mismatch")
        new_cursor = self._token_cursors[layer_id] + int(raw_logits.shape[0])
        if new_cursor > self._expected_tokens:
            raise RuntimeError(
                f"layer {layer_id} observed {new_cursor} tokens, expected {self._expected_tokens}"
            )
        return spec

    def route(
        self,
        *,
        layer_id: int,
        raw_logits: torch.Tensor,
        original_probs: torch.Tensor,
        original_ids: torch.Tensor,
        original_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self._assert_router_inputs(
            layer_id,
            raw_logits,
            original_probs,
            original_ids,
            original_scores,
        )
        device = original_ids.device
        score_dtype = original_scores.dtype

        # A single layer-level transfer avoids a GPU synchronization for every
        # token. This harness optimizes trustworthiness, not routing latency.
        raw_cpu = raw_logits.detach().float().cpu()
        probs_cpu = original_probs.detach().float().cpu()
        orig_ids_cpu = original_ids.detach().cpu()
        orig_scores_cpu = original_scores.detach().float().cpu()

        selected_ids_rows: list[torch.Tensor] = []
        selected_score_rows: list[torch.Tensor] = []
        buffer = self._buffers[layer_id]
        cache = self._caches[layer_id]
        estimator = self._range_means[layer_id]

        for row in range(raw_cpu.shape[0]):
            logits = raw_cpu[row]
            probabilities = probs_cpu[row]
            original_row_ids = orig_ids_cpu[row].to(torch.long)
            original_row_scores = orig_scores_cpu[row]
            delta = float((logits.max() - logits.min()).item())
            delta_mean = estimator.update(delta)

            if self.routing.policy == "original" or self.routing.lambda_value == 0.0:
                selected_ids = original_row_ids.clone()
                selected_scores = original_row_scores.clone()
            elif self.routing.policy == "cache_prior":
                protected = torch.from_numpy(cache.membership(spec.num_experts))
                if self.routing.top_j:
                    protected[original_row_ids[: self.routing.top_j]] = True
                rerank_logits = logits + (
                    self.routing.lambda_value * delta_mean * protected.to(dtype=logits.dtype)
                )
                selected_ids = torch.topk(
                    rerank_logits,
                    k=spec.top_k,
                    dim=-1,
                ).indices
                selected_scores = probabilities.gather(0, selected_ids)
                if spec.norm_topk_prob:
                    selected_scores = selected_scores / selected_scores.sum().clamp_min(
                        torch.finfo(selected_scores.dtype).tiny
                    )
            else:
                raise AssertionError(f"unsupported routing policy {self.routing.policy}")

            ids_list = [int(value) for value in selected_ids.tolist()]
            scores_list = [float(value) for value in selected_scores.tolist()]
            access = cache.observe(ids_list, scores_list)

            buffer.original_ids.append(original_row_ids.numpy().copy())
            buffer.selected_ids.append(selected_ids.numpy().copy())
            buffer.selected_weights.append(selected_scores.numpy().copy())
            buffer.hit_mask.append(np.asarray(access.hits, dtype=np.bool_))
            buffer.logit_range.append(delta)
            buffer.range_mean.append(delta_mean)
            selected_ids_rows.append(selected_ids)
            selected_score_rows.append(selected_scores)

        self._token_cursors[layer_id] += int(raw_cpu.shape[0])
        selected_ids_tensor = torch.stack(selected_ids_rows).to(
            device=device,
            dtype=original_ids.dtype,
        )
        selected_scores_tensor = torch.stack(selected_score_rows).to(
            device=device,
            dtype=score_dtype,
        )
        return selected_scores_tensor, selected_ids_tensor

    def end_sequence(self) -> RouteTrace:
        if self._active_sample is None:
            raise RuntimeError("no active sequence")
        for spec in self.layer_specs:
            observed = self._token_cursors[spec.layer_id]
            if observed != self._expected_tokens:
                raise RuntimeError(
                    f"layer {spec.layer_id} observed {observed} tokens, "
                    f"expected {self._expected_tokens}"
                )

        def stack(field: str) -> np.ndarray:
            return np.stack(
                [
                    np.stack(getattr(self._buffers[spec.layer_id], field), axis=0)
                    for spec in self.layer_specs
                ],
                axis=0,
            )

        trace = RouteTrace(
            sample_id=self._active_sample,
            original_ids=stack("original_ids"),
            selected_ids=stack("selected_ids"),
            selected_weights=stack("selected_weights"),
            hit_mask=stack("hit_mask"),
            logit_range=np.stack(
                [
                    np.asarray(
                        self._buffers[spec.layer_id].logit_range,
                        dtype=np.float32,
                    )
                    for spec in self.layer_specs
                ],
                axis=0,
            ),
            range_mean=np.stack(
                [
                    np.asarray(
                        self._buffers[spec.layer_id].range_mean,
                        dtype=np.float32,
                    )
                    for spec in self.layer_specs
                ],
                axis=0,
            ),
        )
        trace.validate()
        self._active_sample = None
        self._expected_tokens = 0
        self._buffers = {}
        self._token_cursors = {}
        return trace

    def abort_sequence(self) -> None:
        self._active_sample = None
        self._expected_tokens = 0
        self._buffers = {}
        self._token_cursors = {}
        for cache in self._caches.values():
            cache.reset()
