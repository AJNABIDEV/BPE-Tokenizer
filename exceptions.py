class TokenizerError(Exception):
    """Base exception for all tokenizer-related errors."""


class TrainingError(TokenizerError):
    """Raised when training fails or receives invalid input."""


class EncodingError(TokenizerError):
    """Raised when encoding fails."""


class DecodingError(TokenizerError):
    """Raised when decoding fails or a token id is unknown."""


class SerializationError(TokenizerError):
    """Raised when saving or loading tokenizer artifacts fails."""


class CacheError(TokenizerError):
    """Raised when cache read/write operations fail."""


class VocabularyError(TokenizerError):
    """Raised when vocabulary construction or lookup fails."""
