import json
from dataclasses import replace
from pathlib import Path

import torch
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
