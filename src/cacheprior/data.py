from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from cacheprior.config import DatasetConfig


@dataclass(frozen=True)
class TokenWindow:
    dataset_id: str
    sample_id: str
    input_ids: torch.Tensor
    target_ids: torch.Tensor

    @property
    def scored_tokens(self) -> int:
        return int(self.target_ids.numel())

    @property
    def token_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.input_ids.numpy().tobytes())
        digest.update(self.target_ids.numpy().tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class DatasetManifest:
    source: str
    subset: str | None
    split: str
    revision: str | None
    fingerprint: str | None
    mode: str
    prediction_length: int
    max_windows: int | None
    streaming: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def window_token_ids(
    token_ids: Sequence[int],
    prediction_length: int,
) -> Iterator[tuple[list[int], list[int]]]:
    """Create contiguous S-token prediction windows using S+1 source tokens."""
    if prediction_length <= 0:
        raise ValueError("prediction_length must be positive")
    start = 0
    required = prediction_length + 1
    while start + required <= len(token_ids):
        chunk = token_ids[start : start + required]
        yield list(chunk[:-1]), list(chunk[1:])
        start += prediction_length


class HFTextDataset:
    def __init__(self, config: DatasetConfig, tokenizer: Any) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self._dataset: Any | None = None

    def _load(self) -> Any:
        if self._dataset is None:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise RuntimeError(
                    "Hugging Face Datasets is required for real dataset runs. "
                    "Install the project dependencies first."
                ) from exc
            self._dataset = load_dataset(
                self.config.source,
                self.config.subset,
                split=self.config.split,
                revision=self.config.revision,
                streaming=self.config.streaming,
            )
        return self._dataset

    @property
    def dataset_id(self) -> str:
        parts = [self.config.source]
        if self.config.subset:
            parts.append(self.config.subset)
        parts.append(self.config.split)
        return "/".join(parts)

    def manifest(self) -> DatasetManifest:
        dataset = self._load()
        return DatasetManifest(
            source=self.config.source,
            subset=self.config.subset,
            split=self.config.split,
            revision=self.config.revision,
            fingerprint=getattr(dataset, "_fingerprint", None),
            mode=self.config.mode,
            prediction_length=self.config.prediction_length,
            max_windows=self.config.max_windows,
            streaming=self.config.streaming,
        )

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        return [int(value) for value in encoded["input_ids"]]

    def _records(self) -> Iterable[str]:
        dataset = self._load()
        for row in dataset:
            if self.config.text_field not in row:
                raise KeyError(
                    f"dataset row does not contain text field {self.config.text_field!r}"
                )
            text = row[self.config.text_field]
            if text is not None and str(text).strip():
                yield str(text)

    def _make_window(
        self,
        sample_index: int,
        input_ids: Sequence[int],
        target_ids: Sequence[int],
    ) -> TokenWindow:
        return TokenWindow(
            dataset_id=self.dataset_id,
            sample_id=f"{sample_index:06d}",
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            target_ids=torch.tensor([target_ids], dtype=torch.long),
        )

    def _iter_concatenated(self) -> Iterator[TokenWindow]:
        length = self.config.prediction_length
        separator_ids = self._encode(self.config.separator)
        buffer: list[int] = []
        emitted = 0
        for text in self._records():
            if buffer and separator_ids:
                buffer.extend(separator_ids)
            buffer.extend(self._encode(text))
            while len(buffer) >= length + 1:
                source = buffer[: length + 1]
                yield self._make_window(emitted, source[:-1], source[1:])
                emitted += 1
                if self.config.max_windows is not None and emitted >= self.config.max_windows:
                    return
                del buffer[:length]

    def _iter_documents(self) -> Iterator[TokenWindow]:
        emitted = 0
        for text in self._records():
            ids = self._encode(text)
            for input_ids, target_ids in window_token_ids(
                ids,
                self.config.prediction_length,
            ):
                yield self._make_window(emitted, input_ids, target_ids)
                emitted += 1
                if self.config.max_windows is not None and emitted >= self.config.max_windows:
                    return

    def __iter__(self) -> Iterator[TokenWindow]:
        if self.config.mode == "concatenate":
            yield from self._iter_concatenated()
        elif self.config.mode == "document":
            yield from self._iter_documents()
        else:
            raise AssertionError(f"unsupported dataset mode {self.config.mode}")
