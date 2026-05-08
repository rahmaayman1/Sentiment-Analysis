import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import os

def load_data():
    train = pd.read_csv("data/processed/train.csv")
    val   = pd.read_csv("data/processed/val.csv")
    test  = pd.read_csv("data/processed/test.csv")
    return train, val, test

def train_baseline(train, val, test):
    X_train, y_train = train['review'], train['label']
    X_val,   y_val   = val['review'],   val['label']
    X_test,  y_test  = test['review'],  test['label']

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf   = tfidf.transform(X_val)
    X_test_tfidf  = tfidf.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    print("=== Validation ===")
    val_preds = model.predict(X_val_tfidf)
    print(classification_report(y_val, val_preds, target_names=['negative','positive']))

    print("=== Test ===")
    test_preds = model.predict(X_test_tfidf)
    print(classification_report(y_test, test_preds, target_names=['negative','positive']))

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['negative','positive'])
    disp.plot(cmap='Blues')
    plt.title("Baseline — Confusion Matrix")
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/baseline_confusion_matrix.png", dpi=150)
    plt.show()

    # Save
    os.makedirs("models/baseline", exist_ok=True)
    joblib.dump(model, "models/baseline/lr_model.pkl")
    joblib.dump(tfidf, "models/baseline/tfidf.pkl")
    print(" Model saved")

if __name__ == "__main__":
    train, val, test = load_data()
    train_baseline(train, val, test)