from __future__ import annotations

import gc
from collections import defaultdict

from .exceptions import TrainingError
from .utils import Word, get_logger, iter_file_chunks

logger = get_logger(__name__)

_DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


class FrequencyCounter:
    __slots__ = ("_text", "_file_path", "_chunk_size")

    def __init__(self, text: str = "", file_path: str = "", chunk_size: int = _DEFAULT_CHUNK_SIZE) -> None:
        if not text and not file_path:
            raise TrainingError("Provide `text` or `file_path`.")
        self._text = text
        self._file_path = file_path
        self._chunk_size = chunk_size

    @staticmethod
    def _ingest(data: str, freqs: defaultdict[Word, int]) -> None:
        for word in data.split():
            freqs[tuple(word.encode("utf-8"))] += 1

    def count(self) -> dict[Word, int]:
        freqs: defaultdict[Word, int] = defaultdict(int)
        if self._file_path:
            for block in iter_file_chunks(self._file_path, self._chunk_size):
                self._ingest(block, freqs)
            logger.info("%d unique words counted from file", len(freqs))
        else:
            self._ingest(self._text, freqs)
            logger.info("%d unique words counted from text", len(freqs))

        result = dict(freqs)
        del freqs
        gc.collect()
        return result
