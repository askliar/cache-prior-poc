from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RouteTrace:
    sample_id: str
    original_ids: np.ndarray
    selected_ids: np.ndarray
    selected_weights: np.ndarray
    hit_mask: np.ndarray
    logit_range: np.ndarray
    range_mean: np.ndarray

    def validate(self) -> None:
        if self.original_ids.ndim != 3:
            raise ValueError("original_ids must have shape [layers, tokens, top_k]")
        if self.selected_ids.shape != self.original_ids.shape:
            raise ValueError("selected_ids shape mismatch")
        if self.selected_weights.shape != self.original_ids.shape:
            raise ValueError("selected_weights shape mismatch")
        if self.hit_mask.shape != self.original_ids.shape:
            raise ValueError("hit_mask shape mismatch")
        if self.logit_range.shape != self.original_ids.shape[:2]:
            raise ValueError("logit_range shape mismatch")
        if self.range_mean.shape != self.original_ids.shape[:2]:
            raise ValueError("range_mean shape mismatch")

    @property
    def layers(self) -> int:
        return int(self.original_ids.shape[0])

    @property
    def tokens(self) -> int:
        return int(self.original_ids.shape[1])

    @property
    def top_k(self) -> int:
        return int(self.original_ids.shape[2])

    @property
    def accesses(self) -> int:
        return int(self.hit_mask.size)

    @property
    def hits(self) -> int:
        return int(self.hit_mask.sum())

    @property
    def misses(self) -> int:
        return self.accesses - self.hits

    @property
    def changed_tokens(self) -> int:
        original = np.sort(self.original_ids, axis=-1)
        selected = np.sort(self.selected_ids, axis=-1)
        return int(np.any(original != selected, axis=-1).sum())

    @property
    def route_divergence(self) -> float:
        intersections = np.zeros(self.original_ids.shape[:2], dtype=np.int32)
        for rank in range(self.top_k):
            intersections += np.any(
                self.selected_ids == self.original_ids[..., rank, None],
                axis=-1,
            )
        return float(1.0 - intersections.mean() / self.top_k)


def write_trace(trace: RouteTrace, path: str | Path) -> None:
    trace.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    largest_id = max(trace.original_ids.max(), trace.selected_ids.max())
    id_dtype = np.int16 if largest_id < 32768 else np.int32
    np.savez_compressed(
        path,
        sample_id=np.asarray(trace.sample_id),
        original_ids=trace.original_ids.astype(id_dtype, copy=False),
        selected_ids=trace.selected_ids.astype(id_dtype, copy=False),
        # Keep replay priorities in float32. Rounding nearly equal weights to
        # float16 can change the documented within-token cache update order.
        selected_weights=trace.selected_weights.astype(np.float32, copy=False),
        hit_mask=trace.hit_mask.astype(np.uint8, copy=False),
        logit_range=trace.logit_range.astype(np.float32, copy=False),
        range_mean=trace.range_mean.astype(np.float32, copy=False),
    )


def read_trace(path: str | Path) -> RouteTrace:
    with np.load(Path(path), allow_pickle=False) as data:
        trace = RouteTrace(
            sample_id=str(data["sample_id"].item()),
            original_ids=data["original_ids"].astype(np.int64),
            selected_ids=data["selected_ids"].astype(np.int64),
            selected_weights=data["selected_weights"].astype(np.float32),
            hit_mask=data["hit_mask"].astype(np.bool_),
            logit_range=data["logit_range"].astype(np.float32),
            range_mean=data["range_mean"].astype(np.float32),
        )
    trace.validate()
    return trace
