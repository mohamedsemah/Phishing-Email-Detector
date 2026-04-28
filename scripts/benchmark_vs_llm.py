"""
Optional benchmark: compare the trained model vs. a simple TF-IDF + LR baseline on the same test set.
Outputs accuracy, F1, precision, recall for both so you can document "our model outperforms baseline."
No LLM API calls (avoids API keys); add optional LLM comparison separately if desired.
"""
import argparse
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.load_and_merge import load_and_merge, get_train_val_test_splits, get_dataset_dir


def run_baseline(train_texts, train_labels, test_texts, test_labels):
    vectorizer = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)
    return {
        "accuracy": accuracy_score(test_labels, preds),
        "f1": f1_score(test_labels, preds, average="binary", zero_division=0),
        "precision": precision_score(test_labels, preds, average="binary", zero_division=0),
        "recall": recall_score(test_labels, preds, average="binary", zero_division=0),
    }


def run_trained_model(test_texts, test_labels, model_dir):
    from src.api.predict import PhishingPredictor
    predictor = PhishingPredictor(model_dir)
    preds = []
    for t in test_texts:
        out = predictor.predict(t)
        preds.append(1 if out["label"] == "phishing" else 0)
    return {
        "accuracy": accuracy_score(test_labels, preds),
        "f1": f1_score(test_labels, preds, average="binary", zero_division=0),
        "precision": precision_score(test_labels, preds, average="binary", zero_division=0),
        "recall": recall_score(test_labels, preds, average="binary", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark trained model vs TF-IDF+LR baseline")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to saved transformer model")
    parser.add_argument("--max_test", type=int, default=2000, help="Max test samples (for speed)")
    args = parser.parse_args()

    data_dir = get_dataset_dir()
    print("Loading data...")
    df = load_and_merge(data_dir)
    train_df, val_df, test_df = get_train_val_test_splits(df)
    if args.max_test and len(test_df) > args.max_test:
        test_df = test_df.sample(n=args.max_test, random_state=42)
    train_texts = train_df["text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print("Running TF-IDF + Logistic Regression baseline...")
    baseline = run_baseline(train_texts, train_labels, test_texts, test_labels)

    model_dir = args.model_dir or (Path(__file__).resolve().parents[1] / "models" / "distilbert-base-uncased")
    report = [
        "# PRISM benchmark (Predictive Risk Indicator Scoring Model)",
        "",
        f"Test samples: {len(test_labels)}",
        "",
        "## TF-IDF + Logistic Regression (baseline)",
        f"- Accuracy:  {baseline['accuracy']:.4f}",
        f"- F1:        {baseline['f1']:.4f}",
        f"- Precision: {baseline['precision']:.4f}",
        f"- Recall:    {baseline['recall']:.4f}",
        "",
    ]

    if model_dir.exists():
        print("Running trained transformer model...")
        trained = run_trained_model(test_texts, test_labels, model_dir)
        report.extend([
            "## Trained transformer (our model)",
            f"- Accuracy:  {trained['accuracy']:.4f}",
            f"- F1:        {trained['f1']:.4f}",
            f"- Precision: {trained['precision']:.4f}",
            f"- Recall:    {trained['recall']:.4f}",
            "",
        ])
        if trained["accuracy"] >= baseline["accuracy"]:
            report.append("Our model meets or exceeds baseline accuracy on this test set.")
        else:
            report.append("Baseline is higher on this run; try full training (no --max_samples) for better model.")
    else:
        report.append(f"(Trained model not found at {model_dir}; run training first to compare.)")

    out = "\n".join(report)
    print(out)
    out_path = Path(__file__).resolve().parents[1] / "benchmark_report.md"
    out_path.write_text(out, encoding="utf-8")
    print(f"\nReport written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
