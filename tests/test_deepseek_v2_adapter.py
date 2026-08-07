import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from cacheprior.config import CacheConfig, RoutingConfig
from cacheprior.models.deepseek_v2 import DeepseekV2Adapter
from cacheprior.routing import RoutingController


class _FakeGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.top_k = 2
        self.n_routed_experts = 4
        self.norm_topk_prob = False
        self.gating_dim = 4
        self.routed_scaling_factor = 1.0
        self.weight = nn.Parameter(torch.randn(4, 4))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        logits = F.linear(
            hidden_states.reshape(-1, self.gating_dim).float(),
            self.weight.float(),
        )
        probabilities = torch.softmax(logits, dim=-1)
        scores, ids = torch.topk(probabilities, self.top_k, dim=-1, sorted=True)
        return ids.flip(-1), scores.flip(-1), None


class _FakeExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class _FakeMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _FakeGate()
        self.experts = nn.ModuleList([_FakeExpert() for _ in range(4)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, hidden_states.shape[-1])
        ids, scores, _ = self.gate(hidden_states)
        output = torch.zeros_like(flattened)
        for rank in range(ids.shape[-1]):
            for expert_id, expert in enumerate(self.experts):
                mask = ids[:, rank] == expert_id
                if mask.any():
                    output[mask] += expert(flattened[mask]) * scores[mask, rank, None]
        return output.reshape(original_shape)


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _FakeMoe()


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeBackbone()


def test_deepseek_original_wrapper_preserves_outputs_and_ranks_top_j() -> None:
    torch.manual_seed(13)
    model = _FakeModel()
    hidden_states = torch.randn(1, 5, 4)
    with torch.inference_mode():
        baseline = model.model.layers[0].mlp(hidden_states)

    adapter = DeepseekV2Adapter(model)
    controller = RoutingController(
        adapter.layer_specs,
        RoutingConfig(policy="original", top_j=1),
        CacheConfig(capacity=2),
    )
    adapter.install(controller)
    controller.begin_sequence("sample", expected_tokens=hidden_states.shape[1])
    with torch.inference_mode():
        instrumented = model.model.layers[0].mlp(hidden_states)
        model.model.layers[1].mlp(hidden_states)
    trace = controller.end_sequence()
    adapter.restore()

    torch.testing.assert_close(instrumented, baseline, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(trace.original_ids, trace.selected_ids)
    assert adapter.expert_parameter_counts() == {0: 16, 1: 16}
