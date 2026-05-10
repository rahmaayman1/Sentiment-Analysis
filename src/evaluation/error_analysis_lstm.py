import sys
sys.path.append(".")

import torch
import numpy as np
import pandas as pd
import os

from src.models.lstm_model import LSTMModel

# ===== Config =====
VOCAB_SIZE  = 20002
EMBED_DIM   = 128
HIDDEN_DIM  = 256
N_LAYERS    = 2
DROPOUT     = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_error_analysis():
    # ===== Load Data =====
    X_test  = np.load("data/processed/X_test.npy")
    test_df = pd.read_csv("data/processed/test.csv")

    # ===== Load Model =====
    model = LSTMModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load("models/lstm/best_lstm.pt", map_location=DEVICE))
    model.eval()

    # ===== Predictions =====
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            batch  = torch.tensor(X_test[i:i+64], dtype=torch.long).to(DEVICE)
            logits = model(batch)
            probs  = torch.sigmoid(logits)
            preds  = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_df['predicted']      = [int(p)            for p in all_preds]
    test_df['confidence']     = [round(float(p), 4) for p in all_probs]
    test_df['correct']        = test_df['predicted'] == test_df['label']

    label_map = {0: 'negative', 1: 'positive'}
    test_df['label_name']     = test_df['label'].map(label_map)
    test_df['predicted_name'] = test_df['predicted'].map(label_map)

    errors    = test_df[~test_df['correct']].copy()
    top_errors = errors.sort_values('confidence', ascending=False).head(10)

    # ===== Save =====
    os.makedirs("reports/lstm", exist_ok=True)

    with open("reports/lstm/error_analysis.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("ERROR ANALYSIS — LSTM\n")
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
        f.write("COMPARISON WITH RNN\n")
        f.write("=" * 60 + "\n")
        f.write(f"  RNN  Accuracy : 0.5057  ← Vanishing gradient\n")
        f.write(f"  LSTM Accuracy : {test_df['correct'].mean():.4f}  ← Gates solve the problem\n")

    print(" Saved to reports/lstm/error_analysis.txt")

if __name__ == "__main__":
    run_error_analysis()