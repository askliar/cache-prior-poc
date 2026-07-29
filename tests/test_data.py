from cacheprior.data import window_token_ids


def test_windowing_scores_every_token_once_except_final_incomplete_tail() -> None:
    windows = list(window_token_ids(list(range(11)), prediction_length=4))
    assert windows == [
        ([0, 1, 2, 3], [1, 2, 3, 4]),
        ([4, 5, 6, 7], [5, 6, 7, 8]),
    ]


def test_windowing_rejects_invalid_length() -> None:
    try:
        list(window_token_ids([1, 2, 3], prediction_length=0))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")
