# Sentiment Analysis

A complete NLP pipeline for sentiment classification, comparing classical machine learning and deep learning approaches on the IMDB movie reviews dataset.

---

## Overview

This project builds and evaluates four models of increasing complexity to understand the tradeoffs between simplicity, performance, and architectural design in sentiment analysis.

| Model                  | Test Accuracy | Notes                          |
|------------------------|---------------|--------------------------------|
| Baseline (LR + TF-IDF) | 90.00%        | Simple but strong              |
| RNN                    | 50.57%        | Vanishing gradient problem     |
| Bi-LSTM                | 85.80%        | Gates solve vanishing gradient |
| DistilBERT (fine-tuned)| 93.94%        | Best overall performance       |

---

## Project Structure

```
sentiment-analysis/
│
├── data/
│   ├── raw/                        <- Original dataset
│   └── processed/                  <- Train, val, test splits
│
├── models/
│   ├── baseline/                   <- Logistic Regression + TF-IDF
│   ├── rnn/                        <- Vanilla RNN weights
│   ├── lstm/                       <- Bi-LSTM weights
│   ├── bert/                       <- DistilBERT fine-tuned weights
│   └── tokenizer/                  <- Custom tokenizer
│
├── reports/
│   ├── baseline/                   <- Baseline results and error analysis
│   ├── rnn/                        <- RNN results and error analysis
│   ├── lstm/                       <- LSTM results and error analysis
│   ├── bert/                       <- BERT results and error analysis
│   └── final_comparison.txt        <- Full model comparison
│
├── src/
│   ├── data/
│   │   ├── preprocess.py           <- Cleaning, encoding, train/val/test split
│   │   ├── tokenizer.py            <- Custom PyTorch tokenizer
│   │   └── dataset.py              <- PyTorch Dataset and DataLoaders
│   │
│   ├── models/
│   │   ├── baseline.py             <- TF-IDF + Logistic Regression
│   │   ├── rnn_model.py            <- RNN architecture
│   │   ├── rnn_trainer.py          <- RNN training loop
│   │   ├── lstm_model.py           <- Bi-LSTM architecture
│   │   ├── lstm_trainer.py         <- Bi-LSTM training loop
│   │   ├── bert_model.py           <- DistilBERT classification head
│   │   └── bert_trainer.py         <- BERT fine-tuning loop
│   │
│   └── evaluation/
│       ├── error_analysis_baseline.py
│       ├── error_analysis_rnn.py
│       ├── error_analysis_lstm.py
│       ├── error_analysis_bert.py
│       └── compare_models.py       <- Final comparison across all models
│
├── predict.py                      <- CLI inference
├── config.yaml                     <- Centralized configuration
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.12.3

```bash
git clone https://github.com/rahmaayman1/Sentiment-Analysis.git
cd Sentiment-Analysis
pip install -r requirements.txt
```

---

## Pipeline

### 1. Preprocess

```bash
python3 src/data/preprocess.py
```

Cleans HTML tags, encodes labels, and splits data into 80/10/10 train/val/test.

### 2. Tokenize

```bash
python3 src/features/tokenizer.py
```

Builds a custom vocabulary from the training set and pads sequences to MAX_LEN=200.

### 3. Train

```bash
# Baseline
python3 src/models/baseline.py

# RNN
python3 src/models/rnn_trainer.py

# Bi-LSTM
python3 src/models/lstm_trainer.py

# DistilBERT (GPU recommended)
python3 src/models/bert_trainer.py
```

### 4. Evaluate

```bash
python3 src/evaluation/error_analysis_baseline.py
python3 src/evaluation/error_analysis_rnn.py
python3 src/evaluation/error_analysis_lstm.py
python3 src/evaluation/error_analysis_bert.py
python3 src/evaluation/compare_models.py
```

### 5. Inference

```bash
python3 predict.py --text "This movie was absolutely amazing!"
```

```
Sentiment  : positive
Confidence : 97.32%
Model      : DistilBERT
```

---

## Deployment

The API is containerized using Docker and served with FastAPI.

### Run with Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

### Endpoints

| Method | Endpoint   | Description          |
|--------|------------|----------------------|
| GET    | /health    | Health check         |
| POST   | /predict   | Predict sentiment    |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely amazing!"}'
```

### Example Response

```json
{
  "sentiment": "positive",
  "confidence": 0.9732,
  "model": "DistilBERT"
}
```
---

## Key Findings

**Baseline vs RNN:**
Vanilla RNN failed on long sequences (MAX_LEN=200) due to the vanishing gradient problem, achieving only 50% accuracy — equivalent to random guessing. This demonstrates a fundamental limitation of vanilla RNNs on long-range dependencies.

**RNN vs Bi-LSTM:**
LSTM's gating mechanism (input, forget, output gates) resolved the vanishing gradient problem. Accuracy jumped from 50% to 85%, confirming that architectural design choices have a larger impact than training duration alone.

**Bi-LSTM vs Baseline:**
Despite being a more complex architecture, Bi-LSTM (85%) underperformed the Baseline (90%). TF-IDF captures global word statistics that LSTM must learn from scratch without pretrained embeddings.

**Baseline vs DistilBERT:**
DistilBERT (93%) outperformed the Baseline (90%) by leveraging pretraining on 3.3 billion words. The model understands context, negation, and sentiment nuance that bag-of-words approaches miss entirely.

---

## Configuration

All paths and BERT hyperparameters are centralized in `config.yaml`:

```yaml
data:
  raw_path       : "data/raw/IMDB Dataset.csv"
  processed_path : "data/processed"

bert:
  model_name  : "distilbert-base-uncased"
  max_len     : 512
  batch_size  : 16
  lr          : 0.00002
  epochs      : 3
  dropout     : 0.3

paths:
  models      : "models"
  reports     : "reports"
```


---

## Dataset

IMDB Movie Reviews — 50,000 reviews (25,000 positive, 25,000 negative).
Perfectly balanced — no oversampling required.

---

## Tech Stack

- Python 3.12.3
- PyTorch
- HuggingFace Transformers
- scikit-learn
- FastAPI
- Docker
- pandas / numpy
- matplotlib