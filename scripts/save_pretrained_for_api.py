"""
Save a pretrained tokenizer + model (with random classification head) to models/distilbert-base-uncased
so the API can start before full training has been run. Run full training to replace with a trained model.
"""
from pathlib import Path

from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = Path(__file__).resolve().parents[1] / "models" / "distilbert-base-uncased"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {MODEL_NAME} and saving to {OUT_DIR} (untrained head)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    tokenizer.save_pretrained(OUT_DIR)
    model.save_pretrained(OUT_DIR)
    print("Done. API can now load this model. Run 'python -m src.train' for a trained model.")

if __name__ == "__main__":
    main()
