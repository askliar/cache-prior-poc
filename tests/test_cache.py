import numpy as np

from cacheprior.cache import LRUCache, replay_belady, replay_lru


def _single_expert_trace(values: list[int]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(values, dtype=np.int64)[None, :, None]
    priorities = np.ones_like(ids, dtype=np.float32)
    return ids, priorities


def test_lru_known_access_string() -> None:
    ids, priorities = _single_expert_trace([1, 2, 3, 1, 2, 3])
    metrics = replay_lru(ids, priorities, capacity=2)
    assert metrics.accesses == 6
    assert metrics.hits == 0
    assert metrics.misses == 6


def test_belady_known_access_string() -> None:
    ids, priorities = _single_expert_trace([1, 2, 3, 1, 2, 3])
    metrics = replay_belady(ids, priorities, capacity=2)
    assert metrics.accesses == 6
    assert metrics.hits == 2
    assert metrics.misses == 4


def test_hits_use_pre_token_snapshot() -> None:
    cache = LRUCache(capacity=2)
    first = cache.observe([0, 1], [0.9, 0.1])
    assert first.hits == (False, False)
    assert cache.state == (1, 0)

    second = cache.observe([0, 2], [0.8, 0.7])
    assert second.hits == (True, False)
    assert second.evictions == (1,)
    assert cache.state == (2, 0)


def test_belady_never_loses_to_lru_on_random_traces() -> None:
    rng = np.random.default_rng(1234)
    for _ in range(100):
        # choice without replacement keeps every token's top-k unique.
        events = np.stack(
            [rng.choice(8, size=3, replace=False) for _ in range(30)],
            axis=0,
        )[None, ...]
        priorities = rng.random(events.shape, dtype=np.float32)
        lru = replay_lru(events, priorities, capacity=4)
        belady = replay_belady(events, priorities, capacity=4)
        assert belady.misses <= lru.misses


def test_layers_are_independent() -> None:
    ids = np.asarray(
        [
            [[0], [0], [0]],
            [[0], [1], [0]],
        ],
        dtype=np.int64,
    )
    priorities = np.ones_like(ids, dtype=np.float32)
    metrics = replay_lru(ids, priorities, capacity=1)
    assert metrics.per_layer_hits == (2, 0)
