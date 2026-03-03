"""
Fine-tune a transformer for binary phishing vs legitimate email classification.
Saves tokenizer and model to models/ for use by the API.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.load_and_merge import get_dataset_dir, load_and_merge, get_train_val_test_splits


def get_default_model_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def build_dataset(tokenizer, texts, labels, max_length: int = 512):
    """Build Hugging Face compatible dataset from lists."""
    from torch.utils.data import Dataset

    class EmailDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, i):
            item = {
                k: torch.tensor(v[i], dtype=torch.long) for k, v in self.encodings.items()
            }
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )
    encodings = {k: v for k, v in encodings.items()}
    return EmailDataset(encodings, list(labels))


def compute_metrics(pred):
    preds = pred.predictions.argmax(axis=1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description="Train phishing email classifier")
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        help="HuggingFace model: roberta-base, distilbert-base-uncased, microsoft/deberta-v3-base",
    )
    parser.add_argument("--max_length", type=int, default=512, help="Max token length per email")
    parser.add_argument("--epochs", type=int, default=3, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device train batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Output dir for checkpoints; default models/<model_name_safe>")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Stop if val loss does not improve for N evals")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to Kaggle-Dataset; default project Kaggle-Dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap train/val size for quick runs (e.g. 5000)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else get_dataset_dir()
    output_dir = args.output_dir
    if not output_dir:
        safe_name = args.model_name.replace("/", "-")
        output_dir = str(get_default_model_dir() / safe_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading and merging CSV data...")
    df = load_and_merge(data_dir)
    train_df, val_df, test_df = get_train_val_test_splits(df)
    if args.max_samples is not None:
        train_df = train_df.sample(n=min(args.max_samples, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(max(500, args.max_samples // 10), len(val_df)), random_state=42)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_ds = build_dataset(
        tokenizer,
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        max_length=args.max_length,
    )
    val_ds = build_dataset(
        tokenizer,
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print("Training...")
    trainer.train()

    # Save final model and tokenizer to output_dir (used by API)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate on test set
    test_ds = build_dataset(
        tokenizer,
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        max_length=args.max_length,
    )
    test_result = trainer.evaluate(test_ds)
    print("Test set metrics:", test_result)

    # Save test set and metrics for optional benchmark script
    metrics_path = output_path / "test_metrics.txt"
    with open(metrics_path, "w") as f:
        for k, v in test_result.items():
            f.write(f"{k}: {v}\n")
    print(f"Model and tokenizer saved to {output_dir}, test metrics to {metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
