# --- bpe_engine.py ---
import heapq
from collections import defaultdict
from Frequency_counter import FrequencyCounter
from datasets import load_dataset
from Save_files import FileSaver
from tqdm import tqdm

class BPEMergeEngine:
    def __init__(self):
        self.merges = {}
        self.next_id = 256
        self.pair_freqs = defaultdict(int)
        self.pair_to_words = defaultdict(set) 
        self.heap = []
        self.word_freqs = {} # word_tuple: count

    def _get_pairs(self, word: tuple[int, ...]):
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    def generate_merges(self, word_freqs: dict[tuple[int, ...], int], num_merges: int):
        self.word_freqs = word_freqs
        
        # 1. Initial count of all pairs
        for word, count in self.word_freqs.items():
            for pair in self._get_pairs(word):
                self.pair_freqs[pair] += count
                self.pair_to_words[pair].add(word)

        # 2. Build initial heap
        for pair, freq in self.pair_freqs.items():
            heapq.heappush(self.heap, (-freq, pair))

        # 3. Main Merge Loop
        for i in tqdm(range(num_merges), desc="Training BPE"): 
            best_pair = None
            
            # Lazy Deletion: Find the real best pair
            while self.heap:
                freq, pair = heapq.heappop(self.heap)
                if -freq == self.pair_freqs.get(pair, 0):
                    best_pair = pair
                    break
            
            if not best_pair or self.pair_freqs[best_pair] == 0:
                print(f"No more merges possible at {i}.")
                break

            if (i + 1) % 500 == 0:
                print(f"\nMerge {i+1}: Best Pair {best_pair} with freq {-freq}")

            # Update merges map
            self.merges[best_pair] = self.next_id
            p0, p1 = best_pair
            
            # 4. Targetted Update: Only update words that contain the pair
            words_to_update = list(self.pair_to_words[best_pair])
            
            for word in words_to_update:
                if word not in self.word_freqs: continue
                
                count = self.word_freqs.pop(word)
                
                # Remove old pairs count
                for pair in self._get_pairs(word):
                    self.pair_freqs[pair] -= count
                    self.pair_to_words[pair].discard(word)
                
                # Perform Merge in the word
                new_word = []
                idx = 0
                while idx < len(word):
                    if idx < len(word) - 1 and word[idx] == p0 and word[idx+1] == p1:
                        new_word.append(self.next_id)
                        idx += 2
                    else:
                        new_word.append(word[idx])
                        idx += 1
                new_word = tuple(new_word)
                
                # Add back to word_freqs
                self.word_freqs[new_word] = self.word_freqs.get(new_word, 0) + count
                
                # Add new pairs count
                for pair in self._get_pairs(new_word):
                    self.pair_freqs[pair] += count
                    self.pair_to_words[pair].add(new_word)
                    heapq.heappush(self.heap, (-self.pair_freqs[pair], pair))
            
            self.next_id += 1
            if (i + 1) % 100 == 0: print(f"Merge {i + 1}/{num_merges} done.")

        return self.merges

# print("Cloud se data stream ho raha hai...")
# dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True, token=True)

# temp_text_file = "tokenizer_training_data.txt"
# target_conversations = 10000 # 10,000 baatein kaafi hain seekhne ke liye

# with open(temp_text_file, "w", encoding="utf-8") as f:
#     for count, sample in enumerate(dataset):
#         if count >= target_conversations:
#             break
            
#         conversations = sample.get('conversations', [])
#         for turn in conversations:
#             # Har line ko text file me likh rahe hain
#             f.write(turn.get('value', '') + "\n")

# print(f"Data '{temp_text_file}' me save ho gaya!")

# ==========================================
# STEP 2: Tokenizer ko us File se Train Karna
# ==========================================
# ... (Purana code)

# STEP 2: Tokenizer ko us File se Train Karna
print("Words ki frequency count kar rahe hain...")
f = r"C:\Users\jatin\OneDrive\Documents\Work_related\train.txt"
FC = FrequencyCounter(file_path=f)
word_freq = FC.count_frequencies()

# --- YAHAN FIX KARO: Convert strings to integers ---
normalized_word_freq = {}
for word_tuple, count in word_freq.items():
    # Saare characters ko integers mein convert kar rahe hain (ASCII/Unicode)
    int_word = tuple(ord(char) for char in word_tuple)
    normalized_word_freq[int_word] = count
# ----------------------------------------------------

print("BPE Merges generate ho rahe hain...")
BPE = BPEMergeEngine()

# Naya normalized data bhejo
WE = BPE.generate_merges(word_freqs=normalized_word_freq, num_merges=20000)

print("[SUCCESS] Tumhara Tokenizer tayyar hai!")