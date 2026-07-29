from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _load_metrics(run: str | Path) -> dict[str, Any]:
    path = Path(run)
    if path.is_dir():
        path = path / "metrics.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_label(metrics: dict[str, Any]) -> str:
    dataset = metrics["dataset"]
    parts = [dataset["source"]]
    if dataset.get("subset"):
        parts.append(dataset["subset"])
    parts.append(dataset["split"])
    return "/".join(parts)


def comparison_rows(runs: Iterable[str | Path]) -> list[dict[str, Any]]:
    loaded = [_load_metrics(run) for run in runs]
    baselines: dict[str, float] = {}
    for metrics in loaded:
        if metrics["routing"]["policy"] == "original":
            baselines[_dataset_label(metrics)] = float(metrics["quality"]["perplexity"])

    rows: list[dict[str, Any]] = []
    for metrics in loaded:
        dataset = _dataset_label(metrics)
        quality = metrics["quality"]
        policy = metrics["routing"]["policy"]
        lambda_value = metrics["routing"].get("lambda_value", 0.0)
        baseline_ppl = baselines.get(dataset)
        delta_ppl = (
            (float(quality["perplexity"]) / baseline_ppl - 1.0) * 100.0 if baseline_ppl else None
        )

        cache_names = ("none", "lru", "belady") if policy == "original" else ("lru",)
        for cache_name in cache_names:
            cache = metrics["cache"].get(cache_name, {})
            rows.append(
                {
                    "dataset": dataset,
                    "routing": policy,
                    "cache": cache_name,
                    "lambda": lambda_value if policy == "cache_prior" else 0.0,
                    "perplexity": quality["perplexity"],
                    "relative_perplexity_percent": delta_ppl,
                    "miss_rate": cache.get("miss_rate"),
                    "misses_per_scored_token": cache.get("misses_per_scored_token"),
                    "estimated_bytes_per_scored_token": cache.get(
                        "estimated_bytes_per_scored_token"
                    ),
                    "route_divergence": metrics["routing_change"]["mean_set_divergence"],
                    "scored_tokens": quality["scored_tokens"],
                    "run_directory": metrics["run_directory"],
                }
            )
    return rows


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summary(runs: Iterable[str | Path], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(runs)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)
        handle.write("\n")

    fieldnames = list(rows[0]) if rows else []
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headers = [
        "Dataset",
        "Routing",
        "Cache",
        "λ",
        "PPL",
        "ΔPPL %",
        "Miss rate",
        "Misses/token",
        "Est. bytes/token",
        "Route divergence",
    ]
    keys = [
        "dataset",
        "routing",
        "cache",
        "lambda",
        "perplexity",
        "relative_perplexity_percent",
        "miss_rate",
        "misses_per_scored_token",
        "estimated_bytes_per_scored_token",
        "route_divergence",
    ]
    lines = [
        "# Cache-Prior comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format(row[key]) for key in keys) + " |")
    lines.extend(
        [
            "",
            "Estimated bytes are derived from expert misses and configured storage "
            "precision; they are not measured transfers or latency.",
            "",
        ]
    )
    report_path = output_dir / "summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _maybe_plot(rows, output_dir)
    return report_path


def _maybe_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["miss_rate"] is not None:
            grouped[row["dataset"]].append(row)
    for dataset, dataset_rows in grouped.items():
        figure, axis = plt.subplots(figsize=(6, 4))
        for row in dataset_rows:
            label = f"{row['routing']}+{row['cache']}"
            if row["routing"] == "cache_prior":
                label += f" λ={row['lambda']:g}"
            axis.scatter(
                row["miss_rate"],
                row["relative_perplexity_percent"],
                label=label,
            )
        axis.set_xlabel("Expert cache miss rate")
        axis.set_ylabel("Relative perplexity increase (%)")
        axis.set_title(dataset)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"{_safe_name(dataset)}.png", dpi=160)
        plt.close(figure)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")
