import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)
from Embeddings.embeeds import WordEmbedding 
from Error_handler import ErrorHandler
from Tokenizer import BPETokenizer
from Save_files import FileSaver

ErrorHandler.setup_hooks()


def ask_yes_no(prompt):
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("[ERROR] Enter y or n.")


def ask_int(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            return int(raw)
        except ValueError:
            print("[ERROR] Enter a whole number.")


def ask_float(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("[ERROR] Enter a number.")


def ask_dir(prompt):
    return input(prompt).strip() or None


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    book_path = os.path.join(base_dir, "test_book.txt")

    if ask_yes_no("Load pre-trained tokenizer merges? (y/n): "):
        merge_dir = ask_dir("Merges directory (Enter for current): ")
        tokenizer = BPETokenizer(file_path=book_path).load(output_dir=merge_dir)
    else:
        num_merges = ask_int("Number of BPE merges (e.g. 500): ")
        tokenizer = BPETokenizer(file_path=book_path).train(num_merges=num_merges)
        if ask_yes_no("Save merges? (y/n): "):
            save_dir = ask_dir("Save directory (Enter for current): ")
            FileSaver.save_merges(tokenizer._merges, output_dir=save_dir)

    corpus = input("Enter text to encode: ").strip()
    encoded_tokens = tokenizer.encode(corpus)
    print(f"\n[ENCODED] {len(encoded_tokens)} tokens. Sample: {encoded_tokens[:10]}\n")

    if ask_yes_no("Save encoded tokens? (y/n): "):
        save_dir = ask_dir("Save directory (Enter for current): ")
        FileSaver.save_encoded_tokens(encoded_tokens, output_dir=save_dir)

    vocab_size = 256 + len(tokenizer._merges)
    embedding_dim = ask_int(f"Embedding dim [vocab_size={vocab_size}] (e.g. 128): ")
    epochs = ask_int("Epochs (e.g. 5): ")
    window_size = ask_int("Window size (e.g. 3): ")
    lr = ask_float("Learning rate (e.g. 0.02): ")

    embedder = WordEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
    embedder.fit(encoded_tokens, epochs=epochs, window_size=window_size, lr=lr)

    if encoded_tokens:
        sample_id = encoded_tokens[0]
        print(f"[RESULT] Token ID {sample_id} — top 5 closest tokens:")
        print(embedder.query(sample_id, top_n=5))

    embedder.ask_save()


if __name__ == "__main__":
    main()