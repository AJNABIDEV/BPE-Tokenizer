# --- frequency_counter.py ---
import re
from collections import defaultdict


class FrequencyCounter:
    """
    Counts word frequencies from in-memory text or a file (chunked).
    Keys are canonical space-separated byte strings: "104 101 108".
    """

    _UNIT_MAP = {"kb": 1024, "mb": 1024 * 1024}

    def __init__(self, text: str = "", file_path: str = "") -> None:
        if not text and not file_path:
            raise ValueError("Provide either `text` or `file_path`.")
        self.text = text
        self.file_path = file_path
        self.word_freqs: dict[str, int] = defaultdict(int)

    @staticmethod
    def _parse_chunk_size(raw: str) -> int | None:
        match = re.match(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$", raw)
        if not match:
            return None
        size_num = int(match.group(1))
        unit = match.group(2).lower()
        if unit not in FrequencyCounter._UNIT_MAP or not (1 <= size_num <= 64):
            return None
        return size_num * FrequencyCounter._UNIT_MAP[unit]

    @staticmethod
    def _prompt_chunk_size() -> int:
        while True:
            raw = input("[INPUT] Chunk size (1-64 MB/KB, e.g. '8 MB'): ").strip()
            result = FrequencyCounter._parse_chunk_size(raw)
            if result is not None:
                print(f"[INFO] Chunk size set to {result:,} bytes.")
                return result
            print("[ERROR] Invalid. Use a number 1-64 followed by KB or MB.")

    def _ingest(self, text_data: str) -> None:
        for word in text_data.split():
            key = " ".join(map(str, bytes(word, "utf-8")))
            self.word_freqs[key] += 1

    def count_frequencies(self) -> dict[str, int]:
        if self.file_path:
            chunk_size = FrequencyCounter._prompt_chunk_size()
            with open(self.file_path, "r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    self._ingest(chunk)
                    print(f"[INFO] Chunk processed — {len(self.word_freqs):,} unique words so far.")
        else:
            print("[INFO] Processing in-memory text...")
            self._ingest(self.text)

        print(f"[INFO] Done — {len(self.word_freqs):,} unique words.")
        return dict(self.word_freqs)