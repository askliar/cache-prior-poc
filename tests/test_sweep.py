import pytest

from cacheprior.sweep import linear_lambda_grid, log_dense_lambda_grid


def test_linear_lambda_grid_has_uniform_spacing_and_exact_endpoints() -> None:
    values = linear_lambda_grid(250)

    assert len(values) == 250
    assert values[0] == 0.0
    assert values[-1] == 1.0
    spacings = [right - left for left, right in zip(values, values[1:], strict=False)]
    assert all(spacing == pytest.approx(1.0 / 249) for spacing in spacings)


def test_linear_lambda_grid_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        linear_lambda_grid(1)


def test_log_dense_lambda_grid_has_exact_endpoints_and_is_monotonic() -> None:
    values = log_dense_lambda_grid(50, min_positive=1e-3)

    assert len(values) == 50
    assert values[0] == 0.0
    assert values[1] == pytest.approx(1e-3)
    assert values[-1] == 1.0
    assert all(left < right for left, right in zip(values, values[1:], strict=False))


def test_log_dense_lambda_grid_concentrates_points_near_zero() -> None:
    values = log_dense_lambda_grid(50, min_positive=1e-3)

    assert sum(value <= 0.1 for value in values) > len(values) // 2


@pytest.mark.parametrize(
    ("points", "min_positive"),
    [
        (1, 1e-3),
        (50, 0.0),
        (50, 1.0),
    ],
)
def test_log_dense_lambda_grid_rejects_invalid_parameters(
    points: int,
    min_positive: float,
) -> None:
    with pytest.raises(ValueError):
        log_dense_lambda_grid(points, min_positive=min_positive)
