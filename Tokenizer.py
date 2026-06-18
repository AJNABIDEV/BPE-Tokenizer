import heapq
from BPE_Engine import BPEMergeEngine
from Frequency_counter import FrequencyCounter
from Save_files import FileSaver


class BPETokenizer:

    def __init__(self, text: str = "", file_path: str = "") -> None:
        if not text and not file_path:
            raise ValueError("Provide `text` or `file_path`.")

        self._merges: dict[tuple[int, int], int] = {}
        self._vocab: dict[int, bytes] = {}
        self._merge_rank: dict[tuple[int, int], int] = {}
        self._source = file_path if file_path else text
        self._is_file = bool(file_path)

    def train(self, num_merges: int = 256) -> "BPETokenizer":
        fc_kw = {"file_path": self._source} if self._is_file else {"text": self._source}
        word_freqs = FrequencyCounter(**fc_kw).count_frequencies()

        engine = BPEMergeEngine()
        raw_merges = engine.generate_merges(word_freqs, num_merges)

        self._merges = {tuple(map(int, k.split())): v for k, v in raw_merges.items()}
        self._merge_rank = {pair: idx for idx, pair in enumerate(self._merges)}
        self._vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self._merges.items():
            self._vocab[idx] = self._vocab[a] + self._vocab[b]

        return self

    def load(self, output_dir: str | None = None) -> "BPETokenizer":
        raw = FileSaver.load_merges(output_dir=output_dir)
        self._merges = {tuple(map(int, k.split())): v for k, v in raw.items()}
        self._merge_rank = {pair: idx for idx, pair in enumerate(self._merges)}
        self._vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in self._merges.items():
            self._vocab[idx] = self._vocab[a] + self._vocab[b]
        return self

    def encode(self, text: str) -> list[int]:
        ids: list[int] = list(text.encode("utf-8"))
        if len(ids) < 2 or not self._merges:
            return ids

        heap: list[tuple[int, int]] = []
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            if pair in self._merge_rank:
                heapq.heappush(heap, (self._merge_rank[pair], i))

        while heap:
            rank, pos = heapq.heappop(heap)
            if pos >= len(ids) - 1:
                continue
            pair = (ids[pos], ids[pos + 1])
            if self._merge_rank.get(pair) != rank:
                continue
            ids[pos:pos + 2] = [self._merges[pair]]
            for cp in (pos - 1, pos):
                if 0 <= cp < len(ids) - 1:
                    cpair = (ids[cp], ids[cp + 1])
                    if cpair in self._merge_rank:
                        heapq.heappush(heap, (self._merge_rank[cpair], cp))

        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join(self._vocab[i] for i in ids).decode("utf-8", errors="replace")