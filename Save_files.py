# --- save_files.py ---
import json
import os
import tempfile
import numpy as np
import pprint

class FileSaver:
    """
    Atomic JSON/text/numpy writes: temp file → os.replace().
    A mid-write crash never produces a corrupted file.

    Embedding storage (two files per save):
        embeddings.npz       ← compressed numpy matrix  (fast, binary)
        embeddings_meta.json ← vocab_size, embedding_dim, merges path
    """

    @staticmethod
    def _resolve(filename, output_dir=None):
        base = output_dir or os.getcwd()

        if os.path.splitext(base)[1]:
            return base

        os.makedirs(base, exist_ok=True)
        return os.path.join(base, filename)

    @staticmethod
    def _atomic_write_json(data: dict | list, path: str) -> None:
        dir_name = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_text(text: str, path: str) -> None:
        dir_name = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def save_vocab(vocab: list, filename: str = "vocab.json", output_dir: str | None = None) -> None:
        path = FileSaver._resolve(filename, output_dir)
        FileSaver._atomic_write_json(vocab, path)
        print(f"[SAVED] vocab → {path}")

    @staticmethod
    def save_word_freqs(word_freqs: dict, filename: str = "word_freqs.json", output_dir: str | None = None) -> None:
        path = FileSaver._resolve(filename, output_dir)
        FileSaver._atomic_write_json({"data": dict(word_freqs)}, path)
        print(f"[SAVED] word_freqs → {path}")

    @staticmethod
    def load_word_freqs(filename: str = "word_freqs.json", output_dir: str | None = None) -> dict:
        path = FileSaver._resolve(filename, output_dir)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("data", raw)


    @staticmethod
    def save_merges_to_bin(merges, filename="token_ids.bin"):

        token_ids = np.array(list(merges.values()), dtype=np.uint32)

        token_ids.tofile(filename)

        print(f"Saved {len(token_ids)} token IDs.")


    @staticmethod
    def save_merges(merges: dict, filename: str = "merges.json", output_dir: str | None = None) -> None:
        with open(filename, "w", encoding="utf-8") as f:
                    formatted_data = pprint.pformat(merges, indent=4)
                    f.write(formatted_data)

    @staticmethod
    def load_merges(filename: str = "merges.json", output_dir: str | None = None) -> dict:
        path = FileSaver._resolve(filename, output_dir)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def save_encoded_tokens(tokens: list, filename: str = "encoded_tokens.json", output_dir: str | None = None) -> None:
        path = FileSaver._resolve(filename, output_dir)
        FileSaver._atomic_write_json(tokens, path)
        print(f"[SAVED] encoded tokens → {path}")

    @staticmethod
    def save_decoded_text(text: str, filename: str = "decoded_output.txt", output_dir: str | None = None) -> None:
        path = FileSaver._resolve(filename, output_dir)
        FileSaver._atomic_write_text(text, path)
        print(f"[SAVED] decoded text → {path}")

    @staticmethod
    def save_embeddings(matrix: np.ndarray, metadata: dict, name: str = "embeddings", output_dir: str | None = None) -> None:
        base = os.path.abspath(output_dir) if output_dir else os.getcwd()
        os.makedirs(base, exist_ok=True)

        npz_path = os.path.join(base, f"{name}.npz")
        meta_path = os.path.join(base, f"{name}_meta.json")

        fd, tmp_npz = tempfile.mkstemp(dir=base, suffix=".tmp.npz")
        os.close(fd)
        try:
            np.savez_compressed(tmp_npz, matrix=matrix.astype(np.float32))
            os.replace(tmp_npz, npz_path)
        except Exception:
            try:
                os.unlink(tmp_npz)
            except OSError:
                pass
            raise

        meta_full = {
            "vocab_size": int(matrix.shape[0]),
            "embedding_dim": int(matrix.shape[1]),
            **metadata,
        }
        FileSaver._atomic_write_json(meta_full, meta_path)

        print(f"[SAVED] embedding matrix → {npz_path}")
        print(f"[SAVED] embedding meta   → {meta_path}")

    @staticmethod
    def load_embeddings(name: str = "embeddings", output_dir: str | None = None) -> tuple[np.ndarray, dict]:
        base = os.path.abspath(output_dir) if output_dir else os.getcwd()
        npz_path = os.path.join(base, f"{name}.npz")
        meta_path = os.path.join(base, f"{name}_meta.json")

        matrix = np.load(npz_path)["matrix"]
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        print(f"[LOADED] {matrix.shape} matrix from {npz_path}")
        return matrix, meta

    @staticmethod
    def update_embeddings(new_matrix: np.ndarray, metadata: dict | None = None, name: str = "embeddings", output_dir: str | None = None) -> None:
        try:
            _, old_meta = FileSaver.load_embeddings(name, output_dir)
        except FileNotFoundError:
            old_meta = {}

        merged_meta = {**old_meta, **(metadata or {})}
        FileSaver.save_embeddings(new_matrix, merged_meta, name, output_dir)
        print(f"[UPDATED] embeddings '{name}' overwritten.")