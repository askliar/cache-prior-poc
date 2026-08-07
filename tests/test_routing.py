import numpy as np
import torch

from cacheprior.config import CacheConfig, RoutingConfig
from cacheprior.routing import LayerRoutingSpec, RoutingController

SPEC = LayerRoutingSpec(layer_id=0, num_experts=4, top_k=2, norm_topk_prob=False)
CACHE = CacheConfig(capacity=2)


def _original_outputs(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(raw.float(), dim=-1)
    scores, ids = torch.topk(probs, k=2, dim=-1)
    return probs, scores, ids


def test_original_policy_is_transparent() -> None:
    raw = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, -1.0, 2.0, 1.0]])
    probs, scores, ids = _original_outputs(raw)
    controller = RoutingController(
        [SPEC],
        RoutingConfig(policy="original", top_j=1),
        CACHE,
    )
    controller.begin_sequence("sample", expected_tokens=2)
    selected_scores, selected_ids = controller.route(
        layer_id=0,
        raw_logits=raw,
        original_probs=probs,
        original_ids=ids,
        original_scores=scores,
    )
    trace = controller.end_sequence()
    torch.testing.assert_close(selected_scores, scores)
    torch.testing.assert_close(selected_ids, ids)
    np.testing.assert_array_equal(trace.selected_ids, trace.original_ids)


def test_lambda_zero_is_transparent() -> None:
    raw = torch.randn(3, 4)
    probs, scores, ids = _original_outputs(raw)
    controller = RoutingController(
        [SPEC],
        RoutingConfig(policy="cache_prior", lambda_value=0.0, top_j=1),
        CACHE,
    )
    controller.begin_sequence("sample", expected_tokens=3)
    selected_scores, selected_ids = controller.route(
        layer_id=0,
        raw_logits=raw,
        original_probs=probs,
        original_ids=ids,
        original_scores=scores,
    )
    controller.end_sequence()
    torch.testing.assert_close(selected_scores, scores)
    torch.testing.assert_close(selected_ids, ids)


def test_cache_prior_promotes_cached_expert_and_uses_original_probability() -> None:
    raw = torch.tensor(
        [
            [2.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 2.0, 1.0],
        ]
    )
    probs, scores, ids = _original_outputs(raw)
    controller = RoutingController(
        [SPEC],
        RoutingConfig(policy="cache_prior", lambda_value=1.0, top_j=1),
        CACHE,
    )
    controller.begin_sequence("sample", expected_tokens=2)
    selected_scores, selected_ids = controller.route(
        layer_id=0,
        raw_logits=raw,
        original_probs=probs,
        original_ids=ids,
        original_scores=scores,
    )
    trace = controller.end_sequence()

    assert selected_ids[1].tolist() == [2, 0]
    torch.testing.assert_close(
        selected_scores[1],
        probs[1, selected_ids[1]],
    )
    assert trace.hit_mask[0, 1].tolist() == [False, True]
    assert trace.route_divergence == 0.25


def test_top_j_is_retained_without_non_finite_rerank_logits(monkeypatch) -> None:
    raw = torch.randn(20, 4)
    probs, scores, ids = _original_outputs(raw)
    controller = RoutingController(
        [SPEC],
        RoutingConfig(policy="cache_prior", lambda_value=1.0, top_j=1),
        CACHE,
    )
    controller.begin_sequence("sample", expected_tokens=20)
    original_topk = torch.topk

    def finite_topk(values, *args, **kwargs):
        assert torch.isfinite(values).all()
        return original_topk(values, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", finite_topk)
    _, selected_ids = controller.route(
        layer_id=0,
        raw_logits=raw,
        original_probs=probs,
        original_ids=ids,
        original_scores=scores,
    )
    controller.end_sequence()
    assert torch.all(torch.any(selected_ids == ids[:, :1], dim=-1))


def test_full_cache_adds_constant_and_preserves_ranking() -> None:
    controller = RoutingController(
        [SPEC],
        RoutingConfig(policy="cache_prior", lambda_value=1.0, top_j=1),
        CacheConfig(capacity=4),
    )
    raw = torch.tensor(
        [
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 1.0, 3.0],
        ]
    )
    probs, scores, ids = _original_outputs(raw)
    controller.begin_sequence("sample", expected_tokens=3)
    _, selected_ids = controller.route(
        layer_id=0,
        raw_logits=raw,
        original_probs=probs,
        original_ids=ids,
        original_scores=scores,
    )
    controller.end_sequence()
    # The cache is not full until after two events; the final event is the
    # invariant under test.
    torch.testing.assert_close(selected_ids[-1], ids[-1])
