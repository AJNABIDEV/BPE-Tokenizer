from __future__ import annotations

import logging
import os
from collections.abc import Iterator

Pair = tuple[int, int]
Word = tuple[int, ...]


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_path(filename: str, output_dir: str | None = None) -> str:
    base = output_dir or os.getcwd()
    if os.path.splitext(base)[1]:
        return base
    ensure_dir(base)
    return os.path.join(base, filename)


def iter_file_chunks(file_path: str, chunk_size: int = 8 * 1024 * 1024) -> Iterator[str]:
    remainder = ""
    with open(file_path, "r", encoding="utf-8") as f:
        while chunk := f.read(chunk_size):
            split_at = max(chunk.rfind(" "), chunk.rfind("\n"))
            if split_at == -1:
                remainder += chunk
                continue
            yield remainder + chunk[:split_at]
            remainder = chunk[split_at + 1:]
    if remainder:
        yield remainder


def build_vocab(merges: dict[Pair, int], base_vocab_size: int = 256) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(base_vocab_size)}
    for (a, b), new_id in sorted(merges.items(), key=lambda kv: kv[1]):
        vocab[new_id] = vocab[a] + vocab[b]
    return vocab
