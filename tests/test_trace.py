from pathlib import Path

import numpy as np

from cacheprior.trace import RouteTrace, read_trace, write_trace


def test_trace_round_trip(tmp_path: Path) -> None:
    original = np.asarray([[[0, 1], [2, 3]]], dtype=np.int64)
    selected = np.asarray([[[0, 1], [2, 0]]], dtype=np.int64)
    trace = RouteTrace(
        sample_id="000001",
        original_ids=original,
        selected_ids=selected,
        selected_weights=np.asarray([[[0.8, 0.1], [0.7, 0.05]]], dtype=np.float32),
        hit_mask=np.asarray([[[False, False], [False, True]]]),
        logit_range=np.asarray([[2.0, 3.0]], dtype=np.float32),
        range_mean=np.asarray([[2.0, 2.5]], dtype=np.float32),
    )
    path = tmp_path / "trace.npz"
    write_trace(trace, path)
    restored = read_trace(path)
    assert restored.sample_id == trace.sample_id
    np.testing.assert_array_equal(restored.original_ids, trace.original_ids)
    np.testing.assert_array_equal(restored.selected_ids, trace.selected_ids)
    np.testing.assert_allclose(restored.selected_weights, trace.selected_weights)
    np.testing.assert_array_equal(restored.hit_mask, trace.hit_mask)
    assert restored.hits == 1
    assert restored.misses == 3
    assert restored.changed_tokens == 1
