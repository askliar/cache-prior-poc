from __future__ import annotations

import math


def linear_lambda_grid(points: int = 50) -> tuple[float, ...]:
    """Return a reproducible grid with uniformly spaced values in [0, 1]."""

    if points < 2:
        raise ValueError("lambda grid requires at least two points")
    denominator = points - 1
    return tuple(index / denominator for index in range(points))


def log_dense_lambda_grid(
    points: int = 50,
    *,
    min_positive: float = 1e-3,
) -> tuple[float, ...]:
    """Return a reproducible [0, 1] grid concentrated near lambda=0."""

    if points < 2:
        raise ValueError("lambda grid requires at least two points")
    if not 0.0 < min_positive < 1.0:
        raise ValueError("min_positive must be in (0, 1)")
    if points == 2:
        return (0.0, 1.0)

    log_min = math.log(min_positive)
    positive_points = points - 1
    values = [
        math.exp(log_min * (1.0 - index / (positive_points - 1)))
        for index in range(positive_points)
    ]
    values[-1] = 1.0
    return (0.0, *values)
