import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super(LSTMModel, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True    
        )
  
        self.dropout    = nn.Dropout(dropout)
        # hidden_dim 
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        embedded              = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embedded)
        last_hidden           = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(last_hidden)).squeeze(1)


if __name__ == "__main__":
    model = LSTMModel(
        vocab_size=20002,
        embed_dim=128,
        hidden_dim=256,
        n_layers=2,
        dropout=0.3
    )

    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.randint(0, 20000, (64, 200))
    out   = model(dummy)
    print(f"Output shape: {out.shape}")