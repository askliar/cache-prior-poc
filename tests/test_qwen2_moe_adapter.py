import numpy as np
import torch
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from cacheprior.config import CacheConfig, RoutingConfig
from cacheprior.models.qwen2_moe import Qwen2MoeAdapter
from cacheprior.routing import RoutingController


def _tiny_model() -> Qwen2MoeForCausalLM:
    torch.manual_seed(11)
    config = Qwen2MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        decoder_sparse_step=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
        norm_topk_prob=False,
        pad_token_id=0,
        eos_token_id=2,
    )
    return Qwen2MoeForCausalLM(config).eval()


def test_qwen_original_wrapper_is_transparent() -> None:
    model = _tiny_model()
    input_ids = torch.tensor([[1, 4, 7, 9, 3, 2, 5]])
    with torch.inference_mode():
        baseline = model(input_ids=input_ids, use_cache=False).logits

    adapter = Qwen2MoeAdapter(model)
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
    assert adapter.model_type == "qwen2_moe"
    assert trace.layers == 2
    np.testing.assert_array_equal(trace.original_ids, trace.selected_ids)


def test_qwen_cache_prior_retains_protected_expert() -> None:
    model = _tiny_model()
    input_ids = torch.tensor([[1, 4, 7, 9, 3, 2, 5]])
    adapter = Qwen2MoeAdapter(model)
    controller = RoutingController(
        adapter.layer_specs,
        RoutingConfig(policy="cache_prior", lambda_value=0.8, top_j=1),
        CacheConfig(capacity=2),
    )
    adapter.install(controller)
    controller.begin_sequence("sample", expected_tokens=input_ids.shape[1])
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    trace = controller.end_sequence()
    adapter.restore()

    protected = trace.original_ids[..., :1]
    retained = np.any(trace.selected_ids[..., None, :] == protected[..., :, None], axis=-1)
    assert retained.all()
