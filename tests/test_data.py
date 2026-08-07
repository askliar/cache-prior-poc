from cacheprior.config import DatasetConfig
from cacheprior.data import HFTextDataset, window_token_ids


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


def test_join_before_tokenization_builds_one_exact_text_blob() -> None:
    class RecordingTokenizer:
        def __init__(self) -> None:
            self.texts: list[str] = []

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            self.texts.append(text)
            return {"input_ids": [ord(character) for character in text]}

    tokenizer = RecordingTokenizer()
    dataset = HFTextDataset(
        DatasetConfig(
            source="fake",
            mode="concatenate",
            separator="|",
            join_before_tokenization=True,
            prediction_length=2,
            max_windows=1,
        ),
        tokenizer,
    )
    dataset._dataset = [{"text": "ab"}, {"text": ""}, {"text": "cd"}]

    windows = list(dataset)
    assert tokenizer.texts == ["ab||cd"]
    assert windows[0].input_ids.tolist() == [[ord("a"), ord("b")]]
    assert windows[0].target_ids.tolist() == [[ord("b"), ord("|")]]
