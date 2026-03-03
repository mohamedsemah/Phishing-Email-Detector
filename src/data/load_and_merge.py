"""
Load and merge the 7 Kaggle CSV files into a unified (text, label) dataset.
Handles different schemas: text_combined, subject+body, or full metadata.
"""
import os
from pathlib import Path

import pandas as pd


def get_dataset_dir() -> Path:
    """Return path to Kaggle-Dataset directory (project root / Kaggle-Dataset)."""
    return Path(__file__).resolve().parents[2] / "Kaggle-Dataset"


def _safe_str(x) -> str:
    if pd.isna(x) or x is None:
        return ""
    return str(x).strip()


def load_phishing_email(path: Path) -> pd.DataFrame:
    """text_combined, label"""
    df = pd.read_csv(path, usecols=["text_combined", "label"])
    df = df.rename(columns={"text_combined": "text"})
    return df[["text", "label"]]


def load_subject_body(path: Path, cols_subject: str = "subject", cols_body: str = "body") -> pd.DataFrame:
    """subject, body, label -> text = subject + ' ' + body"""
    df = pd.read_csv(path, usecols=[cols_subject, cols_body, "label"], on_bad_lines="skip")
    s = df[cols_subject].fillna("").astype(str).str.strip()
    b = df[cols_body].fillna("").astype(str).str.strip()
    df["text"] = (s + " " + b).str.strip()
    return df[["text", "label"]]


def load_full_schema(path: Path) -> pd.DataFrame:
    """sender, receiver, date, subject, body, label (urls optional) -> rich text"""
    usecols = ["subject", "body", "label"]
    optional = ["sender", "receiver", "urls"]
    try:
        df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    except Exception:
        df = pd.read_csv(path, on_bad_lines="skip", low_memory=False, encoding="utf-8", encoding_errors="ignore")
    available = [c for c in optional if c in df.columns]
    usecols = usecols + available
    df = df[[c for c in usecols if c in df.columns]]
    if "subject" in df.columns and "body" in df.columns:
        df["text"] = "[SUBJECT] " + df["subject"].fillna("").astype(str).str.strip() + " [BODY] " + df["body"].fillna("").astype(str).str.strip()
        if "sender" in df.columns:
            df["text"] = df["text"] + " [SENDER] " + df["sender"].fillna("").astype(str).str.strip()
        if "urls" in df.columns:
            df["text"] = df["text"] + " [URLS] " + df["urls"].fillna("").astype(str).str.strip()
    else:
        df["text"] = df.iloc[:, 0].fillna("").astype(str)
    if "label" not in df.columns:
        df["label"] = 0
    return df[["text", "label"]]


def load_and_merge(data_dir: Path | None = None) -> pd.DataFrame:
    """
    Load all 7 CSVs, unify to (text, label), merge, dedupe, drop empty.
    Returns DataFrame with columns: text, label (0=legitimate, 1=phishing).
    """
    data_dir = data_dir or get_dataset_dir()

    loaders = [
        ("phishing_email.csv", load_phishing_email),
        ("Enron.csv", lambda p: load_subject_body(p)),
        ("Ling.csv", lambda p: load_subject_body(p)),
        ("SpamAssasin.csv", load_full_schema),
        ("Nigerian_Fraud.csv", load_full_schema),
        ("CEAS_08.csv", load_full_schema),
        ("Nazario.csv", load_full_schema),
    ]

    frames = []
    for name, loader in loaders:
        path = data_dir / name
        if not path.exists():
            continue
        try:
            df = loader(path)
            df["source"] = name
            frames.append(df)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["text", "label"]]
    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]
    combined = combined.drop_duplicates(subset=["text"], keep="first")
    combined = combined.reset_index(drop=True)
    combined["label"] = combined["label"].astype(int).clip(0, 1)
    return combined


def get_train_val_test_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
):
    """Stratified train/val/test split. Returns (train_df, val_df, test_df)."""
    from sklearn.model_selection import train_test_split

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train, rest = train_test_split(
        df, test_size=(1 - train_ratio), stratify=df["label"], random_state=random_state
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        rest, test_size=(1 - val_frac), stratify=rest["label"], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    data_dir = get_dataset_dir()
    df = load_and_merge(data_dir)
    print(f"Total samples: {len(df)}, label counts:\n{df['label'].value_counts()}")
    train, val, test = get_train_val_test_splits(df)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
