import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, get_scheduler
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(".")
from src.models.bert_model import BertSentimentModel

import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ===== Config =====
MODEL_NAME    = config['bert']['model_name']
MAX_LEN       = config['bert']['max_len']
BATCH_SIZE    = config['bert']['batch_size']
LR            = config['bert']['lr']
EPOCHS        = config['bert']['epochs']
DROPOUT       = config['bert']['dropout']
PROCESSED_DIR = config['data']['processed_path']
MODEL_DIR     = config['paths']['models']
REPORT_DIR    = config['paths']['reports']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

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
            max_length      = MAX_LEN,
            padding         = 'max_length',
            truncation      = True,
            return_tensors  = 'pt'
        )
        return {
            'input_ids'      : encoded['input_ids'].squeeze(0),
            'attention_mask' : encoded['attention_mask'].squeeze(0),
            'label'          : torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def get_dataloaders(tokenizer):
    train = pd.read_csv(f"{PROCESSED_DIR}/train.csv")
    val   = pd.read_csv(f"{PROCESSED_DIR}/val.csv")
    test  = pd.read_csv(f"{PROCESSED_DIR}/test.csv")

    train_loader = DataLoader(
        BertDataset(train['review'].tolist(), train['label'].tolist(), tokenizer),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        BertDataset(val['review'].tolist(), val['label'].tolist(), tokenizer),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        BertDataset(test['review'].tolist(), test['label'].tolist(), tokenizer),
        batch_size=BATCH_SIZE
    )

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ===== Train one epoch =====
def train_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss, correct = 0, 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        preds = model(input_ids, attention_mask)
        loss  = criterion(preds, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct    += ((preds > 0).float() == labels).sum().item()

    return total_loss / len(loader), correct / (len(loader) * loader.batch_size)


# ===== Evaluate =====
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            preds      = model(input_ids, attention_mask)
            loss       = criterion(preds, labels)
            total_loss += loss.item()
            correct    += ((preds > 0).float() == labels).sum().item()

    return total_loss / len(loader), correct / (len(loader) * loader.batch_size)


# ===== Main =====
def main():
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    train_loader, val_loader, test_loader = get_dataloaders(tokenizer)

    model     = BertSentimentModel(model_name=MODEL_NAME, dropout=DROPOUT).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_scheduler(
        "linear",
        optimizer         = optimizer,
        num_warmup_steps  = total_steps // 10,
        num_training_steps= total_steps
    )

    best_val_acc = 0
    history      = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler)
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
            os.makedirs(f"{MODEL_DIR}/bert", exist_ok=True)
            torch.save(model.state_dict(), f"{MODEL_DIR}/bert/best_bert.pt")
            print(f"   Best model saved (val_acc: {val_acc:.4f})")

    # Test
    model.load_state_dict(torch.load(f"{MODEL_DIR}/bert/best_bert.pt"))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n=== Test Accuracy: {test_acc:.4f} ===")

    # Save Report
    os.makedirs(f"{REPORT_DIR}/bert", exist_ok=True)
    with open(f"{REPORT_DIR}/bert/bert_results.txt", "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BERT TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Config:\n")
        f.write(f"  MODEL      : {MODEL_NAME}\n")
        f.write(f"  MAX_LEN    : {MAX_LEN}\n")
        f.write(f"  BATCH_SIZE : {BATCH_SIZE}\n")
        f.write(f"  LR         : {LR}\n")
        f.write(f"  EPOCHS     : {EPOCHS}\n")
        f.write(f"  DROPOUT    : {DROPOUT}\n\n")

        f.write("=" * 60 + "\n")
        f.write("Training History:\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc':<14}{'Val Loss':<14}{'Val Acc'}\n")
        f.write("-" * 60 + "\n")
        for h in history:
            f.write(f"{h['epoch']:<8}{h['train_loss']:<14}{h['train_acc']:<14}{h['val_loss']:<14}{h['val_acc']}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Final Results:\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Best Val Accuracy : {best_val_acc:.4f}\n")
        f.write(f"  Test Loss         : {test_loss:.4f}\n")
        f.write(f"  Test Accuracy     : {test_acc:.4f}\n")

    print(f" Saved to {REPORT_DIR}/bert/bert_results.txt")


if __name__ == "__main__":
    main()