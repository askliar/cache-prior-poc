from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from cacheprior.cache import (
    combine_replay_metrics,
    replay_belady,
    replay_lru,
    replay_static_prefix,
)
from cacheprior.config import (
    load_dataset_config,
    load_experiment_config,
)
from cacheprior.evaluation import run_experiment
from cacheprior.models.olmoe import load_hf_model_and_tokenizer
from cacheprior.report import write_summary
from cacheprior.sweep import linear_lambda_grid, log_dense_lambda_grid
from cacheprior.trace import read_trace


def _command_validate(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    print(json.dumps(config.to_dict(), indent=2))
    return 0


def _command_run(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    if args.dataset:
        config = config.with_dataset(load_dataset_config(args.dataset))
    if args.routing:
        lambda_value = args.lambda_value
        if args.routing == "original" and lambda_value is None:
            lambda_value = 0.0
        config = config.with_routing(
            policy=args.routing,
            lambda_value=lambda_value,
            top_j=args.top_j,
        )
    elif args.lambda_value is not None:
        config = config.with_routing(
            policy=config.routing.policy,
            lambda_value=args.lambda_value,
        )
    if args.output_root:
        config = replace(
            config,
            trace=replace(config.trace, output_dir=args.output_root),
        )
    run_dir = run_experiment(config)
    print(run_dir)
    return 0


def _trace_paths(path: str | Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    trace_dir = path / "traces" if (path / "traces").is_dir() else path
    paths = sorted(trace_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no NPZ traces found under {path}")
    return paths


def _command_replay(args: argparse.Namespace) -> int:
    metrics = []
    for path in _trace_paths(args.trace):
        trace = read_trace(path)
        ids = trace.original_ids if args.route == "original" else trace.selected_ids
        if args.policy == "lru":
            metrics.append(replay_lru(ids, trace.selected_weights, args.capacity))
        elif args.policy == "belady":
            metrics.append(replay_belady(ids, trace.selected_weights, args.capacity))
        else:
            metrics.append(
                replay_static_prefix(ids, trace.selected_weights, args.capacity)
            )
    result = combine_replay_metrics(metrics).to_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _command_summarize(args: argparse.Namespace) -> int:
    report = write_summary(args.runs, args.output_dir)
    print(report)
    return 0


def _command_matrix(args: argparse.Namespace) -> int:
    base = load_experiment_config(args.base_config)
    if args.top_j is not None:
        base = replace(base, routing=replace(base.routing, top_j=args.top_j))
    if args.output_root:
        base = replace(
            base,
            trace=replace(base.trace, output_dir=args.output_root),
        )
    model, tokenizer = load_hf_model_and_tokenizer(base.model)
    if args.lambdas is not None:
        lambda_values = tuple(args.lambdas)
    elif args.lambda_grid == "log":
        lambda_values = log_dense_lambda_grid(
            args.lambda_points,
            min_positive=args.lambda_min,
        )
    elif args.lambda_grid == "linear":
        lambda_values = linear_lambda_grid(args.lambda_points)
    else:
        lambda_values = (0.5,)
    run_dirs: list[Path] = []
    for dataset_path in args.datasets:
        dataset = load_dataset_config(dataset_path)
        original = base.with_dataset(dataset).with_routing(
            policy="original",
            lambda_value=0.0,
        )
        run_dirs.append(run_experiment(original, model=model, tokenizer=tokenizer))
        for lambda_value in lambda_values:
            cache_prior = base.with_dataset(dataset).with_routing(
                policy="cache_prior",
                lambda_value=lambda_value,
            )
            run_dirs.append(run_experiment(cache_prior, model=model, tokenizer=tokenizer))
    output_dir = Path(args.summary_dir)
    report = write_summary(run_dirs, output_dir)
    print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cacheprior",
        description="Cache-Prior MoE routing proof-of-concept",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    validate.set_defaults(func=_command_validate)

    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset")
    run.add_argument("--routing", choices=("original", "cache_prior"))
    run.add_argument("--lambda", dest="lambda_value", type=float)
    run.add_argument("--top-j", type=int)
    run.add_argument("--output-root")
    run.set_defaults(func=_command_run)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--capacity", required=True, type=int)
    replay.add_argument(
        "--policy", choices=("lru", "belady", "static-prefix"), required=True
    )
    replay.add_argument("--route", choices=("original", "selected"), default="original")
    replay.set_defaults(func=_command_replay)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--runs", nargs="+", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.set_defaults(func=_command_summarize)

    matrix = subparsers.add_parser("matrix")
    matrix.add_argument("--base-config", required=True)
    matrix.add_argument("--datasets", nargs="+", required=True)
    lambda_selection = matrix.add_mutually_exclusive_group()
    lambda_selection.add_argument("--lambdas", nargs="+", type=float)
    lambda_selection.add_argument("--lambda-grid", choices=("log", "linear"))
    matrix.add_argument("--lambda-points", type=int, default=50)
    matrix.add_argument("--lambda-min", type=float, default=1e-3)
    matrix.add_argument("--top-j", type=int)
    matrix.add_argument("--output-root")
    matrix.add_argument("--summary-dir", required=True)
    matrix.set_defaults(func=_command_matrix)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    raise SystemExit(args.func(args))
