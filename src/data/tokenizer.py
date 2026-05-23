import pandas as pd
import numpy as np
from collections import Counter
import pickle
import os

# ===== Config =====
MAX_WORDS = 20000
MAX_LEN   = 200

# ===== Tokenizer Class =====
class SimpleTokenizer:
    def __init__(self, max_words):
        self.max_words = max_words
        self.word2idx  = {"<PAD>": 0, "<OOV>": 1}
        self.idx2word  = {0: "<PAD>", 1: "<OOV>"}

    def fit(self, texts):
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        most_common = counter.most_common(self.max_words - 2)
        for i, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = i
            self.idx2word[i]    = word

    def encode(self, text):
        return [
            self.word2idx.get(word, 1)   
            for word in text.lower().split()
        ]

    def __len__(self):
        return len(self.word2idx)


def pad_sequence(seq, max_len):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [0] * (max_len - len(seq))   


def tokenize_and_pad(tokenizer, texts):
    return np.array([
        pad_sequence(tokenizer.encode(text), MAX_LEN)
        for text in texts
    ], dtype=np.int64)


def main():
    train = pd.read_csv("data/processed/train.csv")
    val   = pd.read_csv("data/processed/val.csv")
    test  = pd.read_csv("data/processed/test.csv")

    tokenizer = SimpleTokenizer(max_words=MAX_WORDS)
    tokenizer.fit(train['review'])

    print(f"Vocabulary size : {len(tokenizer)}")
    print(f"MAX_LEN         : {MAX_LEN}")
    print(f"MAX_WORDS       : {MAX_WORDS}")

    X_train = tokenize_and_pad(tokenizer, train['review'])
    X_val   = tokenize_and_pad(tokenizer, val['review'])
    X_test  = tokenize_and_pad(tokenizer, test['review'])

    y_train = train['label'].values.astype(np.int64)
    y_val   = val['label'].values.astype(np.int64)
    y_test  = test['label'].values.astype(np.int64)

    print(f"\nX_train shape : {X_train.shape}")
    print(f"X_val shape   : {X_val.shape}")
    print(f"X_test shape  : {X_test.shape}")

    # Save
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models/tokenizer", exist_ok=True)

    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_val.npy",   X_val)
    np.save("data/processed/X_test.npy",  X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy",   y_val)
    np.save("data/processed/y_test.npy",  y_test)

    with open("models/tokenizer/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("\n Saved tokenized data to data/processed/")
    print(" Saved tokenizer to models/tokenizer/tokenizer.pkl")


if __name__ == "__main__":
    main()