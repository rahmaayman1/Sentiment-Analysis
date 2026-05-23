import sys
import argparse
import torch
from transformers import DistilBertTokenizer
sys.path.append(".")

from src.models.bert_model import BertSentimentModel

# ===== Config =====
MAX_LEN    = 512
DROPOUT    = 0.3
MODEL_PATH = "models/bert/best_bert.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = BertSentimentModel(dropout=DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def predict(text: str, model, tokenizer) -> dict:
    encoded = tokenizer(
        text,
        max_length     = MAX_LEN,
        padding        = 'max_length',
        truncation     = True,
        return_tensors = 'pt'
    )

    input_ids      = encoded['input_ids'].to(DEVICE)
    attention_mask = encoded['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits     = model(input_ids, attention_mask)
        confidence = torch.sigmoid(logits).item()

    sentiment = "positive" if confidence > 0.5 else "negative"

    score     = confidence if sentiment == "positive" else 1 - confidence

    return {
        "sentiment"  : sentiment,
        "confidence" : round(score, 4),
        "model"      : "DistilBERT"
    }


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis — CLI")
    parser.add_argument("--text", type=str, required=True, help="Input text for sentiment prediction")
    args = parser.parse_args()

    print("Loading model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model     = load_model()
    print(" Model loaded\n")

    result = predict(args.text, model, tokenizer)

    print("=" * 40)
    print(f"Text       : {args.text[:80]}")
    print(f"Sentiment  : {result['sentiment']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"Model      : {result['model']}")
    print("=" * 40)


if __name__ == "__main__":
    main()