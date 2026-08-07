import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import OlmoeConfig, OlmoeForCausalLM

import cacheprior.evaluation as evaluation
from cacheprior.config import (
    CacheConfig,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    RoutingConfig,
    TraceConfig,
)
from cacheprior.data import DatasetManifest, TokenWindow
from cacheprior.report import comparison_rows, write_summary


class _FakeDataset:
    def __init__(self, config: DatasetConfig, tokenizer: object) -> None:
        self.config = config

    def manifest(self) -> DatasetManifest:
        return DatasetManifest(
            source=self.config.source,
            subset=self.config.subset,
            split=self.config.split,
            revision=self.config.revision,
            fingerprint="fake-fingerprint",
            mode=self.config.mode,
            join_before_tokenization=self.config.join_before_tokenization,
            prediction_length=self.config.prediction_length,
            max_windows=1,
            streaming=False,
        )

    def __iter__(self):
        yield TokenWindow(
            dataset_id="fake/text/validation",
            sample_id="000000",
            input_ids=torch.tensor([[1, 3, 5, 7]]),
            target_ids=torch.tensor([[3, 5, 7, 2]]),
        )


def _tiny_model() -> OlmoeForCausalLM:
    torch.manual_seed(11)
    return OlmoeForCausalLM(
        OlmoeConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=8,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=16,
            num_experts=4,
            num_experts_per_tok=2,
            pad_token_id=0,
            eos_token_id=2,
        )
    ).eval()


class _TinyDenseCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, 8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.config = SimpleNamespace(_commit_hash=None)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        del use_cache
        return SimpleNamespace(logits=self.lm_head(self.embed(input_ids)))


def test_original_run_writes_quality_lru_belady_and_trace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(evaluation, "HFTextDataset", _FakeDataset)
    config = ExperimentConfig(
        model=ModelConfig(id="tiny", device="cpu", dtype="float32"),
        dataset=DatasetConfig(
            source="fake/text",
            prediction_length=4,
            max_windows=1,
        ),
        routing=RoutingConfig(policy="original", top_j=1),
        cache=CacheConfig(capacity=2),
        trace=TraceConfig(output_dir=str(tmp_path)),
    )
    model = _tiny_model()
    run_dir = evaluation.run_experiment(config, model=model, tokenizer=object())

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics["cache"]) == {"none", "lru", "belady"}
    assert metrics["quality"]["scored_tokens"] == 4
    assert metrics["windows"] == 1
    assert metrics["routing_change"]["mean_set_divergence"] == 0.0
    assert len(list((run_dir / "traces").glob("*.npz"))) == 1
    assert metrics["cache"]["belady"]["misses"] <= metrics["cache"]["lru"]["misses"]

    # A Cache-Prior config can reuse the same output root without colliding.
    cache_prior = replace(
        config,
        routing=RoutingConfig(
            policy="cache_prior",
            lambda_value=0.5,
            top_j=1,
        ),
    )
    prior_run = evaluation.run_experiment(
        cache_prior,
        model=model,
        tokenizer=object(),
    )
    prior_metrics = json.loads((prior_run / "metrics.json").read_text())
    assert set(prior_metrics["cache"]) == {"none", "lru"}

    rows = comparison_rows([run_dir, prior_run])
    assert [(row["routing"], row["cache"]) for row in rows] == [
        ("original", "none"),
        ("original", "lru"),
        ("original", "belady"),
        ("cache_prior", "lru"),
    ]
    report = write_summary([run_dir, prior_run], tmp_path / "summary")
    assert report.is_file()
    assert (report.parent / "summary.json").is_file()
    assert (report.parent / "summary.csv").is_file()


def test_dense_run_writes_ppl_without_cache_traces(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(evaluation, "HFTextDataset", _FakeDataset)
    config = ExperimentConfig(
        model=ModelConfig(
            id="tiny-dense",
            adapter="dense",
            device="cpu",
            dtype="float32",
        ),
        dataset=DatasetConfig(
            source="fake/text",
            prediction_length=4,
            max_windows=1,
        ),
        routing=RoutingConfig(policy="original", top_j=0),
        cache=CacheConfig(capacity=1),
        trace=TraceConfig(output_dir=str(tmp_path)),
    )

    run_dir = evaluation.run_experiment(
        config,
        model=_TinyDenseCausalLM().eval(),
        tokenizer=object(),
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["model"]["adapter"] == "dense"
    assert metrics["quality"]["scored_tokens"] == 4
    assert metrics["windows"] == 1
    assert set(metrics["cache"]) == {"none"}
    assert not metrics["cache"]["none"]["applicable"]
    assert not metrics["routing_change"]["applicable"]
    assert not list((run_dir / "traces").glob("*.npz"))

    sample = json.loads((run_dir / "per_window.jsonl").read_text())
    assert sample["cache_hits"] is None
    assert sample["trace"] is None


def test_dense_run_rejects_cache_prior(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(evaluation, "HFTextDataset", _FakeDataset)
    config = ExperimentConfig(
        model=ModelConfig(
            id="tiny-dense",
            adapter="dense",
            device="cpu",
            dtype="float32",
        ),
        dataset=DatasetConfig(source="fake/text", prediction_length=4, max_windows=1),
        routing=RoutingConfig(policy="cache_prior", lambda_value=0.5, top_j=0),
        cache=CacheConfig(capacity=1),
        trace=TraceConfig(output_dir=str(tmp_path)),
    )

    with pytest.raises(ValueError, match="dense models"):
        evaluation.run_experiment(
            config,
            model=_TinyDenseCausalLM().eval(),
            tokenizer=object(),
        )
