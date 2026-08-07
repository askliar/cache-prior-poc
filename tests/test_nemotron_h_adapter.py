from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from cacheprior.config import CacheConfig, ModelConfig, RoutingConfig
from cacheprior.models.nemotron_h import NemotronHAdapter
from cacheprior.models.olmoe import load_hf_model_and_tokenizer
from cacheprior.routing import RoutingController


class _FakeNemotronHGate(nn.Module):
    def __init__(self, *, n_group: int = 1) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=4,
            moe_intermediate_size=4,
            mlp_bias=False,
        )
        self.top_k = 2
        self.n_routed_experts = 4
        self.routed_scaling_factor = 2.5
        self.n_group = n_group
        self.topk_group = 1
        self.norm_topk_prob = True
        self.weight = nn.Parameter(torch.randn(4, 4))
        self.register_buffer(
            "e_score_correction_bias",
            torch.tensor([0.0, 0.02, -0.01, 0.01]),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = hidden_states.reshape(-1, self.config.hidden_size)
        logits = F.linear(flattened.float(), self.weight.float())
        scores = logits.sigmoid()
        ids = torch.topk(
            scores + self.e_score_correction_bias,
            self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        weights = scores.gather(-1, ids)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return ids, weights * self.routed_scaling_factor


class _FakeExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(hidden_states)).square())


class _FakeMoe(nn.Module):
    def __init__(self, *, n_group: int = 1) -> None:
        super().__init__()
        self.gate = _FakeNemotronHGate(n_group=n_group)
        self.experts = nn.ModuleList([_FakeExpert() for _ in range(4)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
        ids, weights = self.gate(hidden_states)
        output = torch.zeros_like(flattened)
        for rank in range(ids.shape[-1]):
            for expert_id, expert in enumerate(self.experts):
                mask = ids[:, rank] == expert_id
                if mask.any():
                    output[mask] += expert(flattened[mask]) * weights[mask, rank, None]
        return output.reshape(original_shape)


class _FakeLayer(nn.Module):
    def __init__(self, mixer: nn.Module) -> None:
        super().__init__()
        self.mixer = mixer


class _FakeModel(nn.Module):
    def __init__(self, *, n_group: int = 1) -> None:
        super().__init__()
        self.backbone = SimpleNamespace(
            layers=nn.ModuleList(
                [
                    _FakeLayer(nn.Identity()),
                    _FakeLayer(_FakeMoe(n_group=n_group)),
                ]
            )
        )


class _FakeLoadedModel(nn.Module):
    def to(self, *args, **kwargs):
        raise AssertionError("native packed checkpoint must not be cast with model.to()")


def test_nemotron_original_wrapper_is_transparent() -> None:
    torch.manual_seed(19)
    model = _FakeModel()
    hidden_states = torch.randn(1, 7, 4)
    moe = model.backbone.layers[1].mixer
    with torch.inference_mode():
        baseline = moe(hidden_states)

    adapter = NemotronHAdapter(model)
    controller = RoutingController(
        adapter.layer_specs,
        RoutingConfig(policy="original", top_j=1),
        CacheConfig(capacity=2),
    )
    adapter.install(controller)
    controller.begin_sequence("sample", expected_tokens=hidden_states.shape[1])
    with torch.inference_mode():
        instrumented = moe(hidden_states)
    trace = controller.end_sequence()
    adapter.restore()

    torch.testing.assert_close(instrumented, baseline, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(trace.original_ids, trace.selected_ids)
    assert adapter.layer_specs[0].layer_id == 1
    assert adapter.layer_specs[0].score_scale == 2.5
    assert adapter.expert_parameter_counts() == {1: 32}


def test_nemotron_cache_prior_preserves_scaled_normalized_weights() -> None:
    torch.manual_seed(23)
    model = _FakeModel()
    hidden_states = torch.randn(1, 9, 4)
    adapter = NemotronHAdapter(model)
    controller = RoutingController(
        adapter.layer_specs,
        RoutingConfig(policy="cache_prior", lambda_value=0.7, top_j=1),
        CacheConfig(capacity=2),
    )
    adapter.install(controller)
    controller.begin_sequence("sample", expected_tokens=hidden_states.shape[1])
    with torch.inference_mode():
        model.backbone.layers[1].mixer(hidden_states)
    trace = controller.end_sequence()
    adapter.restore()

    np.testing.assert_allclose(trace.selected_weights.sum(axis=-1), 2.5, rtol=1e-6)
    protected = trace.original_ids[..., :1]
    retained = np.any(trace.selected_ids[..., None, :] == protected[..., :, None], axis=-1)
    assert retained.all()


def test_nemotron_adapter_rejects_grouped_routing() -> None:
    with pytest.raises(ValueError, match="grouped Nemotron-H routing"):
        NemotronHAdapter(_FakeModel(n_group=2))


def test_native_checkpoint_loader_preserves_packed_loading_path(monkeypatch) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    captured: dict[str, object] = {}
    tokenizer = object()
    model = _FakeLoadedModel()

    def fake_tokenizer_loader(*args, **kwargs):
        captured["tokenizer_kwargs"] = kwargs
        return tokenizer

    def fake_model_loader(*args, **kwargs):
        captured["model_kwargs"] = kwargs
        return model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer_loader)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_model_loader)

    loaded_model, loaded_tokenizer = load_hf_model_and_tokenizer(
        ModelConfig(
            id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            adapter="nemotron_h",
            quantization="native",
            trust_remote_code=True,
        )
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    model_kwargs = captured["model_kwargs"]
    assert isinstance(model_kwargs, dict)
    assert model_kwargs["device_map"] == "auto"
    assert "quantization_config" not in model_kwargs
