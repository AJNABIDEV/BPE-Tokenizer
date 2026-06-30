from __future__ import annotations

import gc
import heapq
from collections import defaultdict
import tqdm
from .exceptions import TrainingError
from .utils import Pair, Word, get_logger

logger = get_logger(__name__)


class BPEEngine:
    __slots__ = ("merges", "_next_id")

    def __init__(self, vocab_base_size: int = 256) -> None:
        self.merges: dict[Pair, int] = {}
        self._next_id = vocab_base_size

    @staticmethod
    def _pair_freqs(wf: dict[Word, int]) -> dict[Pair, int]:
        pairs: defaultdict[Pair, int] = defaultdict(int)
        for word, cnt in wf.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += cnt
        return dict(pairs)

    @staticmethod
    def _build_index(wf: dict[Word, int]) -> dict[Pair, set[Word]]:
        idx: defaultdict[Pair, set[Word]] = defaultdict(set)
        for word in wf:
            for i in range(len(word) - 1):
                idx[(word[i], word[i + 1])].add(word)
        return idx

    @staticmethod
    def _merge_incremental(
        wf: dict[Word, int],
        pf: dict[Pair, int],
        idx: dict[Pair, set[Word]],
        pair: Pair,
        new_id: int,
    ) -> set[Pair]:
        p0, p1 = pair
        affected: set[Pair] = set()
        for word in list(idx.get(pair, ())):
            cnt = wf.get(word, 0)
            if not cnt:
                continue
            for j in range(len(word) - 1):
                p = (word[j], word[j + 1])
                pf[p] = pf.get(p, 0) - cnt
                if pf[p] <= 0:
                    pf.pop(p, None)
                idx[p].discard(word)
                affected.add(p)
            out: list[int] = []
            i = 0
            length = len(word)
            while i < length:
                if i < length - 1 and word[i] == p0 and word[i + 1] == p1:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            new_word: Word = tuple(out)
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j + 1])
                pf[p] = pf.get(p, 0) + cnt
                idx[p].add(new_word)
                affected.add(p)
            wf[new_word] = wf.get(new_word, 0) + cnt
            del wf[word]
        return affected

    def train(
        self,
        word_freqs: dict[Word, int],
        num_merges: int,
        min_frequency: int = 2,
        progress_every: int = 10000,
    ) -> dict[Pair, int]:
        if num_merges <= 0:
            raise TrainingError("num_merges must be positive.")

        wf = {k: v for k, v in word_freqs.items() if v >= min_frequency}
        pf = self._pair_freqs(wf)
        if not pf:
            return self.merges

        idx = self._build_index(wf)
        heap = [(-f, p) for p, f in pf.items()]
        heapq.heapify(heap)

        for i in tqdm.tqdm(range(num_merges), desc="Training BPE"):
            best: Pair | None = None
            while heap:
                neg_f, cand = heapq.heappop(heap)
                if pf.get(cand, 0) == -neg_f:
                    best = cand
                    break
            if best is None:
                break

            new_id = self._next_id
            self.merges[best] = new_id
            affected = self._merge_incremental(wf, pf, idx, best, new_id)
            self._next_id += 1

            for p in affected:
                f = pf.get(p, 0)
                if f > 0:
                    heapq.heappush(heap, (-f, p))

            if progress_every and (i + 1) % progress_every == 0:
                logger.info("merge %d/%d, vocab_size=%d", i + 1, num_merges, self._next_id - 1)

        del wf, pf, idx, heap
        gc.collect()
        return self.merges
