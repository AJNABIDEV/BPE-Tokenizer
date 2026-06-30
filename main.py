from .tokenizer import Tokenizer

tok = Tokenizer()
tok.train(file_path=r"C:\Users\jatin\OneDrive\Documents\Work_related\Extra\train.txt", num_merges=100000)
ids = tok.encode("hello world")
text = tok.decode(ids)
tok.save("model")          # writes merges.json, vocab.json, metadata.json
tok.load("model")          # reloads later, no retraining needed