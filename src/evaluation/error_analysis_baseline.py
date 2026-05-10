import pandas as pd
import joblib
from pathlib import Path

# ===== Load =====
test  = pd.read_csv("data/processed/test.csv")
model = joblib.load("models/baseline/lr_model.pkl")
tfidf = joblib.load("models/baseline/tfidf.pkl")

X_test = tfidf.transform(test['review'])
test['predicted']  = model.predict(X_test)
test['confidence'] = model.predict_proba(X_test).max(axis=1)
test['correct']    = test['predicted'] == test['label']

errors = test[~test['correct']].copy()
label_map = {0: 'negative', 1: 'positive'}
errors['label_name']     = errors['label'].map(label_map)
errors['predicted_name'] = errors['predicted'].map(label_map)

feature_names = tfidf.get_feature_names_out()
coefs         = model.coef_[0]

# ===== Save =====
Path("reports/baseline").mkdir(exist_ok=True)

with open("reports/baseline/error_analysis.txt", "w", encoding="utf-8") as f:

    # --- Summary ---
    f.write("=" * 60 + "\n")
    f.write("ERROR ANALYSIS — BASELINE\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total errors : {len(errors)} / {len(test)}\n")
    f.write(f"Accuracy     : {test['correct'].mean():.4f}\n\n")

    breakdown = errors.groupby(['label_name', 'predicted_name']).size()
    f.write("Error breakdown:\n")
    f.write(breakdown.to_string() + "\n\n")

    # --- High confidence mistakes ---
    f.write("=" * 60 + "\n")
    f.write("HIGH CONFIDENCE MISTAKES (top 10)\n")
    f.write("=" * 60 + "\n\n")

    top_errors = errors.sort_values('confidence', ascending=False).head(10)
    for _, row in top_errors.iterrows():
        f.write(f"True      : {row['label_name']}\n")
        f.write(f"Predicted : {row['predicted_name']}\n")
        f.write(f"Confidence: {row['confidence']:.2f}\n")
        f.write(f"Text      : {row['review'][:300]}\n")
        f.write("-" * 60 + "\n\n")

    # --- Top words ---
    f.write("=" * 60 + "\n")
    f.write("TOP WORDS → POSITIVE\n")
    f.write("=" * 60 + "\n")
    for coef, word in sorted(zip(coefs, feature_names), reverse=True)[:15]:
        f.write(f"  {word:30s} {coef:.3f}\n")

    f.write("\n" + "=" * 60 + "\n")
    f.write("TOP WORDS → NEGATIVE\n")
    f.write("=" * 60 + "\n")
    for coef, word in sorted(zip(coefs, feature_names))[:15]:
        f.write(f"  {word:30s} {coef:.3f}\n")

print(" Saved to reports/baseline/error_analysis.txt")