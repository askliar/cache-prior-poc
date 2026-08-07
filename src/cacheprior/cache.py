from __future__ import annotations

from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CacheAccessResult:
    hits: tuple[bool, ...]
    evictions: tuple[int, ...]
    post_state: tuple[int, ...]

    @property
    def misses(self) -> tuple[bool, ...]:
        return tuple(not hit for hit in self.hits)


class LRUCache:
    """Per-layer expert LRU with atomic pre-token hit accounting."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._order: OrderedDict[int, None] = OrderedDict()

    def reset(self) -> None:
        self._order.clear()

    @property
    def state(self) -> tuple[int, ...]:
        """Experts from least to most recent."""
        return tuple(self._order)

    def membership(self, num_experts: int) -> np.ndarray:
        mask = np.zeros(num_experts, dtype=np.bool_)
        if self._order:
            mask[np.fromiter(self._order, dtype=np.int64)] = True
        return mask

    def observe(
        self,
        expert_ids: Sequence[int],
        priorities: Sequence[float],
    ) -> CacheAccessResult:
        if len(expert_ids) != len(priorities):
            raise ValueError("expert_ids and priorities must have equal length")
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("expert IDs within one token must be unique")

        pre_state = set(self._order)
        hits = tuple(int(expert_id) in pre_state for expert_id in expert_ids)
        evictions: list[int] = []

        # Match the paper implementation: insert selected experts from highest
        # to lowest router weight. The highest-weight expert is therefore least
        # recent among this token's accesses and will be evicted first.
        update_order = sorted(
            zip(expert_ids, priorities, strict=True),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
        for expert_id, _ in update_order:
            expert_id = int(expert_id)
            if expert_id in self._order:
                self._order.move_to_end(expert_id)
                continue
            if len(self._order) == self.capacity:
                evicted, _ = self._order.popitem(last=False)
                evictions.append(evicted)
            self._order[expert_id] = None

        return CacheAccessResult(hits, tuple(evictions), self.state)


def vectorized_lru_membership(
    expert_ids: torch.Tensor,
    priorities: torch.Tensor,
    *,
    capacity: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct all pre-token LRU states for a fixed route tensor."""

    if expert_ids.ndim != 2:
        raise ValueError("expert_ids must have shape [tokens, top_k]")
    if priorities.shape != expert_ids.shape:
        raise ValueError("priorities must have the same shape as expert_ids")
    if capacity <= 0 or capacity > num_experts:
        raise ValueError("capacity must be in [1, num_experts]")
    tokens, top_k = expert_ids.shape
    if tokens == 0:
        membership = torch.zeros(
            (0, num_experts),
            dtype=torch.bool,
            device=expert_ids.device,
        )
        return membership, torch.zeros_like(expert_ids, dtype=torch.bool)

    # Highest-priority selected experts are touched first and the lowest last,
    # matching the atomic update order in LRUCache.observe.
    id_order = torch.argsort(expert_ids, dim=-1, stable=True)
    ids_by_id = expert_ids.gather(-1, id_order)
    priorities_by_id = priorities.gather(-1, id_order)
    priority_order = torch.argsort(
        priorities_by_id,
        dim=-1,
        descending=True,
        stable=True,
    )
    ordered_ids = ids_by_id.gather(-1, priority_order).to(torch.long)
    token_offsets = torch.arange(tokens, device=expert_ids.device).unsqueeze(-1) * top_k
    intra_token_order = torch.arange(top_k, device=expert_ids.device).unsqueeze(0)
    timestamps = token_offsets + intra_token_order

    events = torch.full(
        (tokens, num_experts),
        -1,
        dtype=torch.long,
        device=expert_ids.device,
    )
    events.scatter_(1, ordered_ids, timestamps)
    last_use_inclusive = torch.cummax(events, dim=0).values
    last_use_before = torch.cat(
        (
            torch.full_like(last_use_inclusive[:1], -1),
            last_use_inclusive[:-1],
        ),
        dim=0,
    )

    resident_values, resident_ids = torch.topk(
        last_use_before,
        k=capacity,
        dim=-1,
    )
    membership = torch.zeros_like(last_use_before, dtype=torch.bool)
    membership.scatter_(1, resident_ids, resident_values >= 0)
    hits = membership.gather(1, expert_ids.to(torch.long))
    return membership, hits


@dataclass(frozen=True)
class ReplayMetrics:
    accesses: int
    hits: int
    misses: int
    per_layer_accesses: tuple[int, ...]
    per_layer_hits: tuple[int, ...]

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.accesses if self.accesses else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "accesses": self.accesses,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "per_layer_accesses": list(self.per_layer_accesses),
            "per_layer_hits": list(self.per_layer_hits),
            "per_layer_misses": [
                accesses - hits
                for accesses, hits in zip(self.per_layer_accesses, self.per_layer_hits, strict=True)
            ],
        }


def _validate_trace_arrays(
    expert_ids: np.ndarray,
    priorities: np.ndarray,
) -> tuple[int, int, int]:
    if expert_ids.ndim != 3:
        raise ValueError("expert_ids must have shape [layers, tokens, top_k]")
    if priorities.shape != expert_ids.shape:
        raise ValueError("priorities must have the same shape as expert_ids")
    return tuple(int(value) for value in expert_ids.shape)  # type: ignore[return-value]


def replay_lru(
    expert_ids: np.ndarray,
    priorities: np.ndarray,
    capacity: int,
) -> ReplayMetrics:
    layers, tokens, _ = _validate_trace_arrays(expert_ids, priorities)
    layer_accesses: list[int] = []
    layer_hits: list[int] = []
    for layer in range(layers):
        cache = LRUCache(capacity)
        hits = 0
        accesses = 0
        for token in range(tokens):
            ids = expert_ids[layer, token].astype(np.int64).tolist()
            probs = priorities[layer, token].astype(np.float64).tolist()
            result = cache.observe(ids, probs)
            hits += sum(result.hits)
            accesses += len(ids)
        layer_accesses.append(accesses)
        layer_hits.append(hits)
    total_accesses = sum(layer_accesses)
    total_hits = sum(layer_hits)
    return ReplayMetrics(
        total_accesses,
        total_hits,
        total_accesses - total_hits,
        tuple(layer_accesses),
        tuple(layer_hits),
    )


def replay_belady(
    expert_ids: np.ndarray,
    priorities: np.ndarray,
    capacity: int,
) -> ReplayMetrics:
    """Replay Belady on a fixed trace using the same token-event semantics as LRU."""

    layers, tokens, _ = _validate_trace_arrays(expert_ids, priorities)
    layer_accesses: list[int] = []
    layer_hits: list[int] = []

    for layer in range(layers):
        future: dict[int, deque[int]] = defaultdict(deque)
        for token in range(tokens):
            for expert_id in expert_ids[layer, token]:
                future[int(expert_id)].append(token)

        resident: set[int] = set()
        hits = 0
        accesses = 0
        for token in range(tokens):
            ids = [int(value) for value in expert_ids[layer, token]]
            probs = [float(value) for value in priorities[layer, token]]
            pre_state = set(resident)
            hits += sum(expert_id in pre_state for expert_id in ids)
            accesses += len(ids)

            for expert_id in ids:
                queue = future[expert_id]
                if not queue or queue[0] != token:
                    raise AssertionError("invalid Belady future-use queue")
                queue.popleft()

            update_order = sorted(
                zip(ids, probs, strict=True),
                key=lambda item: (-item[1], item[0]),
            )
            for expert_id, _ in update_order:
                if expert_id in resident:
                    continue
                if len(resident) == capacity:

                    def next_use(
                        candidate: int,
                        future_uses: dict[int, deque[int]] = future,
                    ) -> tuple[float, int]:
                        queue = future_uses[candidate]
                        distance = float(queue[0]) if queue else float("inf")
                        return distance, candidate

                    victim = max(resident, key=next_use)
                    resident.remove(victim)
                resident.add(expert_id)

        layer_accesses.append(accesses)
        layer_hits.append(hits)

    total_accesses = sum(layer_accesses)
    total_hits = sum(layer_hits)
    return ReplayMetrics(
        total_accesses,
        total_hits,
        total_accesses - total_hits,
        tuple(layer_accesses),
        tuple(layer_hits),
    )


def replay_static_prefix(
    expert_ids: np.ndarray,
    priorities: np.ndarray,
    capacity: int,
) -> ReplayMetrics:
    """Replay a fixed cache containing expert IDs ``[0, capacity)`` per layer.

    The cache has no warm-up, insertions, evictions, or dependence on the
    access order. ``priorities`` is validated for trace compatibility but does
    not affect the static-cache result.
    """

    layers, _, _ = _validate_trace_arrays(expert_ids, priorities)
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    hit_mask = (expert_ids >= 0) & (expert_ids < capacity)
    layer_accesses = tuple(int(expert_ids[layer].size) for layer in range(layers))
    layer_hits = tuple(int(hit_mask[layer].sum()) for layer in range(layers))
    accesses = sum(layer_accesses)
    hits = sum(layer_hits)
    return ReplayMetrics(
        accesses,
        hits,
        accesses - hits,
        layer_accesses,
        layer_hits,
    )


def combine_replay_metrics(metrics: Iterable[ReplayMetrics]) -> ReplayMetrics:
    metrics = list(metrics)
    if not metrics:
        return ReplayMetrics(0, 0, 0, (), ())
    layer_count = len(metrics[0].per_layer_accesses)
    if any(len(metric.per_layer_accesses) != layer_count for metric in metrics):
        raise ValueError("cannot combine traces with different layer counts")
    per_layer_accesses = tuple(
        sum(metric.per_layer_accesses[layer] for metric in metrics) for layer in range(layer_count)
    )
    per_layer_hits = tuple(
        sum(metric.per_layer_hits[layer] for metric in metrics) for layer in range(layer_count)
    )
    accesses = sum(per_layer_accesses)
    hits = sum(per_layer_hits)
    return ReplayMetrics(
        accesses,
        hits,
        accesses - hits,
        per_layer_accesses,
        per_layer_hits,
    )
