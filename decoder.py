from __future__ import annotations

from .exceptions import DecodingError
from .utils import Pair, build_vocab


class Decoder:
    __slots__ = ("_vocab",)

    def __init__(self, merges: dict[Pair, int], base_vocab_size: int = 256) -> None:
        self._vocab = build_vocab(merges, base_vocab_size)

    def decode(self, ids: list[int]) -> str:
        vocab = self._vocab
        try:
            return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")
        except KeyError as exc:
            raise DecodingError(f"Unknown token id: {exc}") from exc
