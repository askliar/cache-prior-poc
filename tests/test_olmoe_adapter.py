import numpy as np
import torch
from transformers import OlmoeConfig, OlmoeForCausalLM

from cacheprior.config import CacheConfig, RoutingConfig
from cacheprior.models.olmoe import OlmoeAdapter
from cacheprior.routing import RoutingController


def _tiny_model() -> OlmoeForCausalLM:
    torch.manual_seed(7)
    config = OlmoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        num_experts=4,
        num_experts_per_tok=2,
        pad_token_id=0,
        eos_token_id=2,
        norm_topk_prob=False,
    )
    return OlmoeForCausalLM(config).eval()


def test_original_wrapper_is_transparent() -> None:
    model = _tiny_model()
    input_ids = torch.tensor([[1, 4, 7, 9, 3, 2, 5]])
    with torch.inference_mode():
        baseline = model(input_ids=input_ids, use_cache=False).logits

    adapter = OlmoeAdapter(model)
    controller = RoutingController(
        adapter.layer_specs,
        RoutingConfig(policy="original", top_j=1),
        CacheConfig(capacity=2),
    )
    adapter.install(controller)
    controller.begin_sequence("sample", expected_tokens=input_ids.shape[1])
    with torch.inference_mode():
        instrumented = model(input_ids=input_ids, use_cache=False).logits
    trace = controller.end_sequence()
    adapter.restore()

    torch.testing.assert_close(instrumented, baseline, rtol=0, atol=0)
    assert trace.layers == 2
    assert trace.tokens == input_ids.shape[1]
    np.testing.assert_array_equal(trace.original_ids, trace.selected_ids)


def test_full_sequence_matches_token_by_token_with_cache_prior() -> None:
    model = _tiny_model()
    input_ids = torch.tensor([[1, 4, 7, 9, 3, 2]])
    adapter = OlmoeAdapter(model)

    full_controller = RoutingController(
        adapter.layer_specs,
        RoutingConfig(policy="cache_prior", lambda_value=0.5, top_j=1),
        CacheConfig(capacity=2),
    )
    adapter.install(full_controller)
    full_controller.begin_sequence("full", expected_tokens=input_ids.shape[1])
    with torch.inference_mode():
        full_logits = model(input_ids=input_ids, use_cache=False).logits
    full_trace = full_controller.end_sequence()
    adapter.restore()

    incremental_controller = RoutingController(
        adapter.layer_specs,
        RoutingConfig(policy="cache_prior", lambda_value=0.5, top_j=1),
        CacheConfig(capacity=2),
    )
    adapter.install(incremental_controller)
    incremental_controller.begin_sequence(
        "incremental",
        expected_tokens=input_ids.shape[1],
    )
    past_key_values = None
    incremental_logits = []
    with torch.inference_mode():
        for token in range(input_ids.shape[1]):
            output = model(
                input_ids=input_ids[:, token : token + 1],
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            incremental_logits.append(output.logits)
    incremental_trace = incremental_controller.end_sequence()
    adapter.restore()

    torch.testing.assert_close(
        torch.cat(incremental_logits, dim=1),
        full_logits,
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_array_equal(
        incremental_trace.selected_ids,
        full_trace.selected_ids,
    )
