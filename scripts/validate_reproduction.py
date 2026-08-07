#!/usr/bin/env python3
"""Validate that a set of completed experiment runs is internally comparable."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _window_fingerprint(path: Path) -> tuple[tuple[str, str, int], ...]:
    windows: list[tuple[str, str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            windows.append(
                (
                    str(row["sample_id"]),
                    str(row["token_hash"]),
                    int(row["scored_tokens"]),
                )
            )
    return tuple(windows)


def validate_group(
    root: Path,
    expected_lambdas: tuple[float, ...],
) -> dict[str, Any]:
    run_dirs = sorted(path.parent for path in root.glob("*/metrics.json"))
    if not run_dirs:
        raise ValueError(f"no completed runs found under {root}")

    reference_windows: tuple[tuple[str, str, int], ...] | None = None
    original_runs = 0
    observed_lambdas: list[float] = []
    model_ids: set[str] = set()
    dataset_ids: set[tuple[str, str | None, str]] = set()
    scored_tokens: set[int] = set()
    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        metrics = _load_json(run_dir / "metrics.json")
        windows = _window_fingerprint(run_dir / "per_window.jsonl")
        if reference_windows is None:
            reference_windows = windows
        elif windows != reference_windows:
            raise ValueError(f"dataset windows differ in {run_dir}")

        model_ids.add(str(metrics["model"]["model_id"]))
        dataset = metrics["dataset"]
        dataset_ids.add((dataset["source"], dataset.get("subset"), dataset["split"]))
        scored_tokens.add(int(metrics["quality"]["scored_tokens"]))
        policy = str(metrics["routing"]["policy"])
        lambda_value = float(metrics["routing"]["lambda_value"])
        perplexity = float(metrics["quality"]["perplexity"])
        miss_rate = float(metrics["cache"]["lru"]["miss_rate"])
        top_j_retention = float(metrics["routing_change"]["top_j_retention"])
        if not math.isfinite(perplexity) or not math.isfinite(miss_rate):
            raise ValueError(f"non-finite metric in {run_dir}")
        if (
            policy == "cache_prior"
            and int(metrics["routing"]["top_j"]) > 0
            and top_j_retention != 1.0
        ):
            raise ValueError(f"top-J retention is not exact in {run_dir}")
        if policy == "original":
            original_runs += 1
        elif policy == "cache_prior":
            observed_lambdas.append(lambda_value)
        else:
            raise ValueError(f"unexpected routing policy {policy!r} in {run_dir}")
        rows.append(
            {
                "policy": policy,
                "lambda": lambda_value,
                "perplexity": perplexity,
                "miss_rate": miss_rate,
                "top_j_retention": top_j_retention,
            }
        )

    if original_runs != 1:
        raise ValueError(f"expected one original run under {root}, found {original_runs}")
    if sorted(observed_lambdas) != sorted(expected_lambdas):
        raise ValueError(
            f"lambda sweep mismatch under {root}: observed {sorted(observed_lambdas)}, "
            f"expected {sorted(expected_lambdas)}"
        )
    if len(model_ids) != 1 or len(dataset_ids) != 1 or len(scored_tokens) != 1:
        raise ValueError(f"run manifests differ under {root}")

    rows.sort(key=lambda row: (row["policy"] != "original", row["lambda"]))
    return {
        "root": str(root),
        "model": next(iter(model_ids)),
        "dataset": list(next(iter(dataset_ids))),
        "run_count": len(run_dirs),
        "window_count": len(reference_windows or ()),
        "scored_tokens": next(iter(scored_tokens)),
        "identical_token_windows": True,
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument(
        "--expected-lambdas",
        nargs="+",
        type=float,
        default=[value / 10 for value in range(1, 11)],
    )
    args = parser.parse_args()
    expected_lambdas = tuple(args.expected_lambdas)
    result = [
        validate_group(path.expanduser().resolve(), expected_lambdas)
        for path in args.run_roots
    ]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
