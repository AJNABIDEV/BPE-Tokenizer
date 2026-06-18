from Frequency_counter import FrequencyCounter


class VocabMaker:
    def __init__(self, text: str = "", file_path: str = "") -> None:
        if not text and not file_path:
            raise ValueError("Provide `text` or `file_path`.")
        self.text = text
        self.file_path = file_path

    def make_character_vocab(self) -> list[str]:
        if self.file_path:
            chars: set = set()
            chunk = FrequencyCounter.DEFAULT_CHUNK
            with open(self.file_path, "r", encoding="utf-8") as f:
                while data := f.read(chunk):
                    chars.update(data)
            vocab = sorted(chars)
        else:
            vocab = sorted(set(self.text))
        print(f"[VOCAB] {len(vocab)} unique characters.")
        return vocab

    @staticmethod
    def make_bpe_base_vocab() -> dict[int, bytes]:
        return {i: bytes([i]) for i in range(256)}