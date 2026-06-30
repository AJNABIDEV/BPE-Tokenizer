from __future__ import annotations

import json
import os
import tempfile
from typing import Any

from .exceptions import SerializationError
from .utils import Pair, Word, get_logger, resolve_path

logger = get_logger(__name__)

_FORMAT_VERSION = 1


def _atomic_write_json(data: Any, path: str) -> None:
    dir_name = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception as exc:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise SerializationError(f"Failed writing {path}: {exc}") from exc


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SerializationError(f"Failed reading {path}: {exc}") from exc


class Serializer:
    MERGES_FILE = "merges.json"
    VOCAB_FILE = "vocab.json"
    META_FILE = "metadata.json"
    WORD_FREQS_FILE = "word_freqs.json"

    @staticmethod
    def save_word_freqs(word_freqs: dict[Word, int], output_dir: str | None = None) -> str:
        path = resolve_path(Serializer.WORD_FREQS_FILE, output_dir)
        serializable = {" ".join(map(str, k)): v for k, v in word_freqs.items()}
        _atomic_write_json({"data": serializable}, path)
        logger.info("word frequencies saved -> %s", path)
        return path

    @staticmethod
    def load_word_freqs(path: str) -> dict[Word, int]:
        raw = _read_json(path)
        data = raw.get("data", raw)
        return {tuple(map(int, k.split())): v for k, v in data.items()}

    @staticmethod
    def save_merges(merges: dict[Pair, int], output_dir: str | None = None) -> str:
        path = resolve_path(Serializer.MERGES_FILE, output_dir)
        serializable = [[a, b, idx] for (a, b), idx in merges.items()]
        _atomic_write_json(serializable, path)
        logger.info("merges saved -> %s", path)
        return path

    @staticmethod
    def load_merges(output_dir: str | None = None) -> dict[Pair, int]:
        path = resolve_path(Serializer.MERGES_FILE, output_dir)
        raw = _read_json(path)
        return {(a, b): idx for a, b, idx in raw}

    @staticmethod
    def save_vocab(vocab: dict[int, bytes], output_dir: str | None = None) -> str:
        path = resolve_path(Serializer.VOCAB_FILE, output_dir)
        serializable = {str(k): v.hex() for k, v in vocab.items()}
        _atomic_write_json(serializable, path)
        logger.info("vocab saved -> %s", path)
        return path

    @staticmethod
    def load_vocab(output_dir: str | None = None) -> dict[int, bytes]:
        path = resolve_path(Serializer.VOCAB_FILE, output_dir)
        raw = _read_json(path)
        return {int(k): bytes.fromhex(v) for k, v in raw.items()}

    @staticmethod
    def save_metadata(metadata: dict[str, Any], output_dir: str | None = None) -> str:
        path = resolve_path(Serializer.META_FILE, output_dir)
        full = {"format_version": _FORMAT_VERSION, **metadata}
        _atomic_write_json(full, path)
        logger.info("metadata saved -> %s", path)
        return path

    @staticmethod
    def load_metadata(output_dir: str | None = None) -> dict[str, Any]:
        path = resolve_path(Serializer.META_FILE, output_dir)
        return _read_json(path)
