import os

def main():
    results = {
        "Baseline (LR + TF-IDF)" : {"accuracy": 0.9000, "errors": 500,  "note": "Simple but strong"},
        "RNN"                     : {"accuracy": 0.5057, "errors": 2442, "note": "Vanishing gradient"},
        "LSTM"                    : {"accuracy": 0.8580, "errors": 710,  "note": "Gates solve vanishing gradient"},
        "BERT (DistilBERT)"       : {"accuracy": 0.9394, "errors": 303,  "note": "Pretrained transformer"},
    }

    os.makedirs("reports", exist_ok=True)

    with open("reports/final_comparison.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL MODEL COMPARISON — SENTIMENT ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        # Table header
        f.write(f"{'Model':<25}{'Accuracy':<12}{'Errors/5000':<14}{'Note'}\n")
        f.write("-" * 70 + "\n")

        for model, metrics in results.items():
            f.write(
                f"{model:<25}"
                f"{metrics['accuracy']:<12.4f}"
                f"{metrics['errors']:<14}"
                f"{metrics['note']}\n"
            )

        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. Baseline vs RNN:\n")
        f.write("   Vanilla RNN failed completely due to vanishing gradient on\n")
        f.write("   long sequences (MAX_LEN=200). Accuracy stayed at ~50% (random).\n\n")

        f.write("2. RNN vs LSTM:\n")
        f.write("   LSTM's gates (input, forget, output) solved the vanishing\n")
        f.write("   gradient problem. Accuracy jumped from 50% to 85%.\n\n")

        f.write("3. LSTM vs Baseline:\n")
        f.write("   LSTM (85%) still underperforms Baseline (90%) without\n")
        f.write("   pretrained embeddings. TF-IDF captures global word statistics\n")
        f.write("   that LSTM needs more data/epochs to learn from scratch.\n\n")

        f.write("4. Baseline vs BERT:\n")
        f.write("   BERT (93%) outperforms Baseline (90%) by leveraging\n")
        f.write("   pretraining on 3.3B words. Understands context, negation,\n")
        f.write("   and sarcasm better than bag-of-words approaches.\n\n")

        f.write("5. Best model:\n")
        f.write("   BERT (DistilBERT) — 93.94% accuracy, 303 errors only.\n")
        f.write("   Error breakdown is balanced (159 FP, 144 FN) — no class bias.\n\n")

        f.write("=" * 70 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 70 + "\n\n")
        f.write("   Use BERT for production — best accuracy and balanced errors.\n")
        f.write("   Use Baseline for fast inference — 90% accuracy with no GPU needed.\n")

    print(" Saved to reports/final_comparison.txt")

if __name__ == "__main__":
    main()