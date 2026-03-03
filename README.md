# Phishing Email Detector

Web app that detects whether an email is phishing or legitimate. You can **upload a screenshot** of an email (OCR extracts text, then the model predicts) or **paste email text** (subject/body) and get a prediction with confidence.

## Setup

1. **Python 3.10+** and pip.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Model:** A pretrained DistilBERT with a classification head is saved under `models/distilbert-base-uncased` so the API can run immediately. For **better accuracy**, train on the full dataset:
   ```bash
   python -m src.train --model_name roberta-base --epochs 3
   ```
   Or a quicker run with a subset:
   ```bash
   python -m src.train --model_name distilbert-base-uncased --epochs 1 --max_samples 5000
   ```
   Training saves the model under `models/<model_name>/` (e.g. `models/roberta-base`). To use a different path, set env: `PHISHING_MODEL_DIR=/path/to/model`.

## Run the app

From the project root:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser. The first time you use **screenshot upload**, EasyOCR may download language data (one-time).

Screenshot upload supports common image formats like **PNG**, **JPEG**, and **WEBP** by default. **HEIC/HEIF** is supported via `pillow-heif`. AVIF support depends on the installed decoder; if an AVIF file fails, convert it to PNG/JPEG.

You’ll see:

- **Option 1:** Upload a screenshot of an email (image file). The app runs OCR (EasyOCR) and then the phishing model.
- **Option 2:** Paste subject and/or body and click “Analyze pasted text”.

Result: **Phishing** or **Legitimate** plus a confidence score.

## API

- **POST /predict/image** — body: multipart form with `file` (image). Returns `{ "label": "phishing"|"legitimate", "confidence": float, "text_preview": "..." }`.
- **POST /predict/text** — body: JSON `{ "text": "..." }` or `{ "subject": "...", "body": "..." }`. Same response shape.

## Data and training

The dataset is under `Kaggle-Dataset/`: 7 CSV files (phishing vs legitimate emails). The pipeline in `src/data/load_and_merge.py` merges them into a single `(text, label)` dataset and supports stratified train/val/test splits.

- **Train:** `python -m src.train [--model_name roberta-base] [--epochs 3] [--max_samples N]`
- **Benchmark (optional):** Compare the trained model to a TF-IDF + Logistic Regression baseline on the test set:
  ```bash
  python scripts/benchmark_vs_llm.py [--model_dir models/roberta-base] [--max_test 2000]
  ```
  Writes `benchmark_report.md` with accuracy, F1, precision, recall.

## Project layout

- `Kaggle-Dataset/` — 7 CSV files (do not edit).
- `src/data/load_and_merge.py` — load and merge CSVs, stratified split.
- `src/train.py` — fine-tune transformer (RoBERTa/DistilBERT), save to `models/`.
- `src/api/main.py` — FastAPI app (predict/image, predict/text, serve UI).
- `src/api/predict.py` — load model and run inference.
- `src/api/templates/index.html` + `static/app.js` — upload and paste UI.
- `scripts/save_pretrained_for_api.py` — save pretrained model so API works without training.
- `scripts/benchmark_vs_llm.py` — benchmark script vs baseline.
