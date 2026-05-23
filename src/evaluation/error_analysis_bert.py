import sys
sys.path.append(".")

import torch
import pandas as pd
import os
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader

from src.models.bert_model import BertSentimentModel

# ===== Config =====
MAX_LEN    = 512
BATCH_SIZE = 16
DROPOUT    = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Dataset =====
class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length     = MAX_LEN,
            padding        = 'max_length',
            truncation     = True,
            return_tensors = 'pt'
        )
        return {
            'input_ids'      : encoded['input_ids'].squeeze(0),
            'attention_mask' : encoded['attention_mask'].squeeze(0),
            'label'          : torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def run_error_analysis():
    # ===== Load Data =====
    test_df   = pd.read_csv("data/processed/test.csv")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    test_loader = DataLoader(
        BertDataset(test_df['review'].tolist(), test_df['label'].tolist(), tokenizer),
        batch_size=BATCH_SIZE
    )

    # ===== Load Model =====
    model = BertSentimentModel(dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load("models/bert/best_bert.pt", map_location=DEVICE))
    model.eval()

    # ===== Predictions =====
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            logits         = model(input_ids, attention_mask)
            probs          = torch.sigmoid(logits)
            preds          = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_df['predicted']      = [int(p)               for p in all_preds]
    test_df['confidence']     = [round(float(p), 4)   for p in all_probs]
    test_df['correct']        = test_df['predicted']  == test_df['label']

    label_map = {0: 'negative', 1: 'positive'}
    test_df['label_name']     = test_df['label'].map(label_map)
    test_df['predicted_name'] = test_df['predicted'].map(label_map)

    errors     = test_df[~test_df['correct']].copy()
    top_errors = errors.sort_values('confidence', ascending=False).head(10)

    # ===== Save =====
    os.makedirs("reports/bert", exist_ok=True)

    with open("reports/bert/error_analysis.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ERROR ANALYSIS — BERT (DistilBERT)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total test samples : {len(test_df)}\n")
        f.write(f"Total errors       : {len(errors)}\n")
        f.write(f"Accuracy           : {test_df['correct'].mean():.4f}\n\n")

        f.write("=" * 60 + "\n")
        f.write("Error breakdown:\n")
        f.write("=" * 60 + "\n")
        breakdown = errors.groupby(['label_name', 'predicted_name']).size()
        f.write(breakdown.to_string() + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("HIGH CONFIDENCE MISTAKES (top 10)\n")
        f.write("=" * 60 + "\n\n")
        for _, row in top_errors.iterrows():
            f.write(f"True      : {row['label_name']}\n")
            f.write(f"Predicted : {row['predicted_name']}\n")
            f.write(f"Confidence: {row['confidence']:.4f}\n")
            f.write(f"Text      : {row['review'][:300]}\n")
            f.write("-" * 60 + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("FINAL COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Baseline  : 0.9000\n")
        f.write(f"  RNN       : 0.5057  ← Vanishing gradient\n")
        f.write(f"  LSTM      : 0.8580\n")
        f.write(f"  BERT      : {test_df['correct'].mean():.4f}\n")

    print(" Saved to reports/bert/error_analysis.txt")


if __name__ == "__main__":
    run_error_analysis()