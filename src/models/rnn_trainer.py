import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import os
import sys
sys.path.append(".")

from src.data.dataset import get_dataloaders
from src.models.rnn_model import RNNModel

# ===== Config =====
VOCAB_SIZE  = 20002
EMBED_DIM   = 128
HIDDEN_DIM  = 256
N_LAYERS    = 2
DROPOUT     = 0.3
LR          = 1e-3
EPOCHS      = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += ((preds > 0).float() == y_batch).sum().item()

    return total_loss / len(loader), correct / (len(loader) * loader.batch_size)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            preds      = model(X_batch)
            loss       = criterion(preds, y_batch)
            total_loss += loss.item()
            correct    += ((preds > 0).float() == y_batch).sum().item()

    return total_loss / len(loader), correct / (len(loader) * loader.batch_size)

def main():
    train_loader, val_loader, test_loader = get_dataloaders()

    model     = RNNModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        history.append({
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 4),
            "train_acc"  : round(train_acc,  4),
            "val_loss"   : round(val_loss,   4),
            "val_acc"    : round(val_acc,    4),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models/rnn", exist_ok=True)
            torch.save(model.state_dict(), "models/rnn/best_rnn.pt")
            print(f"   Best model saved (val_acc: {val_acc:.4f})")

    # Test
    model.load_state_dict(torch.load("models/rnn/best_rnn.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n=== Test Accuracy: {test_acc:.4f} ===")

    # ===== Save Report =====
    os.makedirs("reports", exist_ok=True)
    with open("reports/rnn_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RNN TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Config:\n")
        f.write(f"  VOCAB_SIZE : {VOCAB_SIZE}\n")
        f.write(f"  EMBED_DIM  : {EMBED_DIM}\n")
        f.write(f"  HIDDEN_DIM : {HIDDEN_DIM}\n")
        f.write(f"  N_LAYERS   : {N_LAYERS}\n")
        f.write(f"  DROPOUT    : {DROPOUT}\n")
        f.write(f"  LR         : {LR}\n")
        f.write(f"  EPOCHS     : {EPOCHS}\n\n")

        f.write("=" * 60 + "\n")
        f.write("Training History:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}{'Val Loss':<14}{'Val Acc':<14}\n")
        f.write("-" * 60 + "\n")
        for h in history:
            f.write(f"{h['epoch']:<8}{h['train_loss']:<14}{h['train_acc']:<14}{h['val_loss']:<14}{h['val_acc']:<14}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Final Results:\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Best Val Accuracy  : {best_val_acc:.4f}\n")
        f.write(f"  Test Loss          : {test_loss:.4f}\n")
        f.write(f"  Test Accuracy      : {test_acc:.4f}\n")

    print("\n Saved to reports/rnn_results.txt")

if __name__ == "__main__":
    main()