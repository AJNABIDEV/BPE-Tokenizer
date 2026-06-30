from __future__ import annotations

import os

from .bpe_engine import BPEEngine
from .cache import Cache
from .exceptions import TrainingError
from .frequency_counter import FrequencyCounter
from .serialization import Serializer
from .utils import Pair, Word, build_vocab, get_logger

logger = get_logger(__name__)


class Trainer:
    __slots__ = ("_cache",)

    def __init__(self, cache_dir: str | None = None) -> None:
        self._cache = Cache(cache_dir)

    def _load_word_freqs(self, text: str, file_path: str, use_cache: bool) -> dict[Word, int]:
        if use_cache:
            cached_path = self._cache.find("word_freqs")
            if cached_path:
                logger.info("word frequency cache found -> %s", cached_path)
                return Serializer.load_word_freqs(cached_path)

        logger.info("counting word frequencies")
        freqs = FrequencyCounter(text=text, file_path=file_path).count()

        if use_cache:
            Serializer.save_word_freqs(freqs, output_dir=self._cache.directory)

        return freqs

    def train(
        self,
        text: str = "",
        file_path: str = "",
        num_merges: int = 1000,
        min_frequency: int = 2,
        use_cache: bool = True,
    ) -> tuple[dict[Pair, int], dict[int, bytes]]:
        if not text and not file_path:
            raise TrainingError("Provide `text` or `file_path`.")
        if file_path and not os.path.isfile(file_path):
            raise TrainingError(f"File not found: {file_path}")

        word_freqs = self._load_word_freqs(text, file_path, use_cache)

        merges = BPEEngine().train(word_freqs, num_merges, min_frequency=min_frequency)
        vocab = build_vocab(merges)

        logger.info("training complete, vocab_size=%d", len(vocab))
        return merges, vocab
