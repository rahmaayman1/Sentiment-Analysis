import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ===== Config =====
BATCH_SIZE = 64

# ===== Dataset Class =====
class IMDBDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders():
    X_train = np.load("data/processed/X_train.npy")
    X_val   = np.load("data/processed/X_val.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val   = np.load("data/processed/y_val.npy")
    y_test  = np.load("data/processed/y_test.npy")

    train_loader = DataLoader(IMDBDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(IMDBDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(IMDBDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    
    X_batch, y_batch = next(iter(train_loader))
    print(f"\nX_batch shape : {X_batch.shape}")
    print(f"y_batch shape : {y_batch.shape}")
    print(f"y_batch sample: {y_batch[:5]}")