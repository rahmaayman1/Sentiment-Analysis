import torch
import torch.nn as nn
from transformers import DistilBertModel

class BertSentimentModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', dropout=0.3):
        super(BertSentimentModel, self).__init__()

        self.bert       = DistilBertModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        output      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token   = output.last_hidden_state[:, 0, :]   
        return self.classifier(self.dropout(cls_token)).squeeze(1)


if __name__ == "__main__":
    model = BertSentimentModel(dropout=0.3)

    print(model)
    print(f"\nTotal parameters    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    input_ids      = torch.randint(0, 1000, (4, 128))
    attention_mask = torch.ones(4, 128, dtype=torch.long)
    out            = model(input_ids, attention_mask)
    print(f"Output shape        : {out.shape}")  