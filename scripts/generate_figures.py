"""
Generate paper-ready figures (PNG) for the phishing email detector project.

Outputs are written to ./figures by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_matplotlib():
    # Import lazily so a clearer error is printed if matplotlib isn't installed.
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required to generate figures. Install it with: pip install matplotlib"
        ) from e


def _savefig(path: Path):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def figure_overall_class_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    counts = df["label"].value_counts().sort_index()
    labels = ["legitimate (0)", "phishing (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=["#2E7D32", "#C62828"])
    plt.ylabel("Number of emails")
    plt.title("Overall class distribution (merged dataset)")
    out_path = out_dir / "fig2_overall_class_distribution.png"
    _savefig(out_path)
    return out_path


def figure_label_distribution_per_source(raw_with_source: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    # raw_with_source expected columns: text, label, source
    src_counts = (
        raw_with_source.groupby(["source", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "legitimate", 1: "phishing"})
    )
    if "legitimate" not in src_counts.columns:
        src_counts["legitimate"] = 0
    if "phishing" not in src_counts.columns:
        src_counts["phishing"] = 0
    src_counts = src_counts[["legitimate", "phishing"]].sort_values(
        by=["phishing", "legitimate"], ascending=False
    )

    x = np.arange(len(src_counts.index))
    width = 0.42

    plt.figure(figsize=(10, 4.5))
    plt.bar(x - width / 2, src_counts["legitimate"].values, width, label="legitimate (0)", color="#2E7D32")
    plt.bar(x + width / 2, src_counts["phishing"].values, width, label="phishing (1)", color="#C62828")
    plt.xticks(x, [str(s).replace(".csv", "") for s in src_counts.index], rotation=20, ha="right")
    plt.ylabel("Number of emails")
    plt.title("Label distribution per dataset source (before deduplication)")
    plt.legend()

    out_path = out_dir / "fig1_label_distribution_per_source.png"
    _savefig(out_path)
    return out_path


def figure_text_length_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    # Approximate lengths by characters (tokenizer-independent).
    lengths = df["text"].fillna("").astype(str).str.len().clip(0, 10_000)
    legit = lengths[df["label"] == 0]
    phish = lengths[df["label"] == 1]

    plt.figure(figsize=(10, 4.5))
    bins = np.linspace(0, 5000, 60)
    plt.hist(legit, bins=bins, alpha=0.65, label="legitimate (0)", color="#2E7D32")
    plt.hist(phish, bins=bins, alpha=0.65, label="phishing (1)", color="#C62828")
    plt.xlabel("Email length (characters, clipped at 10,000)")
    plt.ylabel("Count")
    plt.title("Email text length distribution by class")
    plt.legend()

    out_path = out_dir / "fig_len_text_length_distribution.png"
    _savefig(out_path)
    return out_path


def load_raw_with_source(data_dir: Path) -> pd.DataFrame:
    # Mirrors loader list from src/data/load_and_merge.py but keeps "source".
    from src.data.load_and_merge import load_full_schema, load_phishing_email, load_subject_body

    loaders = [
        ("phishing_email.csv", load_phishing_email),
        ("Enron.csv", lambda p: load_subject_body(p)),
        ("Ling.csv", lambda p: load_subject_body(p)),
        ("SpamAssasin.csv", load_full_schema),
        ("Nigerian_Fraud.csv", load_full_schema),
        ("CEAS_08.csv", load_full_schema),
        ("Nazario.csv", load_full_schema),
    ]

    frames: list[pd.DataFrame] = []
    for name, loader in loaders:
        path = data_dir / name
        if not path.exists():
            continue
        df = loader(path)
        df["source"] = name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    raw = pd.concat(frames, ignore_index=True)
    raw["text"] = raw["text"].fillna("").astype(str).str.strip()
    raw = raw[raw["text"].str.len() > 0].reset_index(drop=True)
    raw["label"] = raw["label"].astype(int).clip(0, 1)
    return raw[["text", "label", "source"]]


def main() -> int:
    _ensure_matplotlib()

    parser = argparse.ArgumentParser(description="Generate PNG charts for the paper")
    parser.add_argument("--out_dir", type=str, default="figures", help="Output directory for PNG figures")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to Kaggle-Dataset directory")
    args = parser.parse_args()

    # Add project root to sys.path so `import src...` works.
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    out_dir = (project_root / args.out_dir).resolve()

    from src.data.load_and_merge import get_dataset_dir, load_and_merge

    data_dir = Path(args.data_dir).resolve() if args.data_dir else get_dataset_dir()
    print(f"Using data_dir: {data_dir}")
    print(f"Writing figures to: {out_dir}")

    raw = load_raw_with_source(data_dir)
    merged = load_and_merge(data_dir)

    paths = [
        figure_label_distribution_per_source(raw, out_dir),
        figure_overall_class_distribution(merged, out_dir),
        figure_text_length_distribution(merged, out_dir),
    ]
    print("Generated:")
    for p in paths:
        print(f"- {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

