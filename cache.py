from __future__ import annotations

import os

from .exceptions import CacheError

_FILENAMES = {
    "word_freqs": "word_freqs.json",
    "merges": "merges.json",
    "vocab": "vocab.json",
}


class Cache:
    __slots__ = ("_dir",)

    def __init__(self, cache_dir: str | None = None) -> None:
        self._dir = os.path.abspath(cache_dir or os.getcwd())

    @property
    def directory(self) -> str:
        return self._dir

    def path_for(self, kind: str) -> str:
        if kind not in _FILENAMES:
            raise CacheError(f"Unknown cache kind: {kind!r}")
        return os.path.join(self._dir, _FILENAMES[kind])

    def exists(self, kind: str) -> bool:
        return os.path.isfile(self.path_for(kind))

    def find(self, kind: str, search_root: str | None = None) -> str | None:
        if kind not in _FILENAMES:
            raise CacheError(f"Unknown cache kind: {kind!r}")
        direct = self.path_for(kind)
        if os.path.isfile(direct):
            return direct
        filename = _FILENAMES[kind]
        root = search_root or self._dir
        for current_dir, _, files in os.walk(root):
            if filename in files:
                return os.path.join(current_dir, filename)
        return None
