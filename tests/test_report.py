import json
from pathlib import Path

from cacheprior.report import comparison_rows


def _write_metrics(
    path: Path,
    *,
    model: str,
    policy: str,
    perplexity: float,
) -> Path:
    metrics = {
        "dataset": {
            "source": "Salesforce/wikitext",
            "subset": "wikitext-2-raw-v1",
            "split": "validation",
        },
        "model": {"model_id": model},
        "routing": {"policy": policy, "lambda_value": 0.5},
        "quality": {"perplexity": perplexity, "scored_tokens": 10},
        "cache": {
            "lru": {
                "miss_rate": 0.2,
                "misses_per_scored_token": 1.0,
                "estimated_bytes_per_scored_token": 2.0,
            },
            "none": {},
            "belady": {
                "miss_rate": 0.1,
                "misses_per_scored_token": 0.5,
                "estimated_bytes_per_scored_token": 1.0,
            },
        },
        "routing_change": {"mean_set_divergence": 0.0},
        "run_directory": str(path.parent),
    }
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return path


def test_comparison_baselines_are_independent_per_model(tmp_path: Path) -> None:
    paths = [
        _write_metrics(
            tmp_path / "a-original.json",
            model="model-a",
            policy="original",
            perplexity=10.0,
        ),
        _write_metrics(
            tmp_path / "a-cache.json",
            model="model-a",
            policy="cache_prior",
            perplexity=11.0,
        ),
        _write_metrics(
            tmp_path / "b-original.json",
            model="model-b",
            policy="original",
            perplexity=20.0,
        ),
        _write_metrics(
            tmp_path / "b-cache.json",
            model="model-b",
            policy="cache_prior",
            perplexity=22.0,
        ),
    ]
    rows = comparison_rows(paths)
    cache_rows = [row for row in rows if row["routing"] == "cache_prior"]
    assert {row["model"] for row in cache_rows} == {"model-a", "model-b"}
    assert all(abs(row["relative_perplexity_percent"] - 10.0) < 1e-12 for row in cache_rows)
    assert [
        (row["model"], row["routing"], row["cache"])
        for row in rows
    ] == [
        ("model-a", "original", "none"),
        ("model-a", "original", "lru"),
        ("model-a", "original", "belady"),
        ("model-a", "cache_prior", "lru"),
        ("model-b", "original", "none"),
        ("model-b", "original", "lru"),
        ("model-b", "original", "belady"),
        ("model-b", "cache_prior", "lru"),
    ]
