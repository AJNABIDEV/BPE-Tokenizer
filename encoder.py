from __future__ import annotations

import heapq

from .exceptions import EncodingError
from .utils import Pair


class Encoder:
    __slots__ = ("_merges",)

    def __init__(self, merges: dict[Pair, int]) -> None:
        self._merges = merges

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise EncodingError("text must be a string")
        if not text:
            return []

        ids: list[int] = list(text.encode("utf-8"))
        n = len(ids)
        merges = self._merges
        if n < 2 or not merges:
            return ids

        prev = list(range(-1, n - 1))
        next_ = list(range(1, n + 1))
        next_[-1] = -1
        alive = bytearray(b"\x01" * n)

        heap: list[tuple[int, int]] = []
        for i in range(n - 1):
            new_id = merges.get((ids[i], ids[i + 1]))
            if new_id is not None:
                heap.append((new_id, i))
        heapq.heapify(heap)

        while heap:
            new_id, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = next_[i]
            if j == -1:
                continue
            cur_id = merges.get((ids[i], ids[j]))
            if cur_id != new_id:
                continue

            ids[i] = new_id
            alive[j] = 0

            k = next_[j]
            next_[i] = k
            if k != -1:
                prev[k] = i

            p = prev[i]
            if p != -1:
                left_id = merges.get((ids[p], ids[i]))
                if left_id is not None:
                    heapq.heappush(heap, (left_id, p))
            if k != -1:
                right_id = merges.get((ids[i], ids[k]))
                if right_id is not None:
                    heapq.heappush(heap, (right_id, i))

        out: list[int] = []
        i = 0
        while i != -1:
            if alive[i]:
                out.append(ids[i])
            i = next_[i]
        return out
