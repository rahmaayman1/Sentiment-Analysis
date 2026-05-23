import sys
sys.path.append(".")

import torch
import yaml
from fastapi import FastAPI
from transformers import DistilBertTokenizer

from src.models.bert_model import BertSentimentModel
from src.api.schemas import PredictRequest, PredictResponse

# ===== Config =====
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_NAME = config['bert']['model_name']
MAX_LEN    = config['bert']['max_len']
DROPOUT    = config['bert']['dropout']
MODEL_PATH = f"{config['paths']['models']}/bert/best_bert.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load Model =====
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-local")
model     = BertSentimentModel(model_name="distilbert-local", dropout=DROPOUT).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===== App =====
app = FastAPI(title="Sentiment Analysis API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    encoded = tokenizer(
        request.text,
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

    return PredictResponse(
        sentiment  = sentiment,
        confidence = round(score, 4),
        model      = "DistilBERT"
    )