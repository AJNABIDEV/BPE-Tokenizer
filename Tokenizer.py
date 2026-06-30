from __future__ import annotations

from .decoder import Decoder
from .encoder import Encoder
from .exceptions import TokenizerError
from .serialization import Serializer
from .trainer import Trainer
from .utils import Pair, build_vocab, get_logger

logger = get_logger(__name__)


class Tokenizer:
    __slots__ = ("_merges", "_vocab", "_encoder", "_decoder")

    def __init__(self) -> None:
        self._merges: dict[Pair, int] = {}
        self._vocab: dict[int, bytes] = {}
        self._encoder: Encoder | None = None
        self._decoder: Decoder | None = None

    def train(
        self,
        file_path: str = "",
        text: str = "",
        num_merges: int = 1000,
        min_frequency: int = 2,
        use_cache: bool = True,
        cache_dir: str | None = None,
    ) -> "Tokenizer":
        merges, vocab = Trainer(cache_dir).train(
            text=text,
            file_path=file_path,
            num_merges=num_merges,
            min_frequency=min_frequency,
            use_cache=use_cache,
        )
        self._merges = merges
        self._vocab = vocab
        self._encoder = Encoder(merges)
        self._decoder = Decoder(merges)
        return self

    def encode(self, text: str) -> list[int]:
        if self._encoder is None:
            raise TokenizerError("Tokenizer not trained or loaded.")
        return self._encoder.encode(text)

    def decode(self, ids: list[int]) -> str:
        if self._decoder is None:
            raise TokenizerError("Tokenizer not trained or loaded.")
        return self._decoder.decode(ids)

    def save(self, output_dir: str = "model") -> None:
        if not self._merges:
            raise TokenizerError("Nothing to save: tokenizer not trained or loaded.")
        Serializer.save_merges(self._merges, output_dir=output_dir)
        Serializer.save_vocab(self._vocab, output_dir=output_dir)
        Serializer.save_metadata(
            {"vocab_size": len(self._vocab), "num_merges": len(self._merges)},
            output_dir=output_dir,
        )
        logger.info("tokenizer saved -> %s", output_dir)

    def load(self, output_dir: str = "model") -> "Tokenizer":
        merges = Serializer.load_merges(output_dir=output_dir)
        self._merges = merges
        self._vocab = build_vocab(merges)
        self._encoder = Encoder(merges)
        self._decoder = Decoder(merges)
        logger.info("tokenizer loaded <- %s", output_dir)
        return self

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)
