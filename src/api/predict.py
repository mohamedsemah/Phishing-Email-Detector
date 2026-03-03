"""
Load the fine-tuned transformer and run inference. Used by both /predict/text and /predict/image (after OCR).
"""
from pathlib import Path

import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABEL_NAMES = ["legitimate", "phishing"]
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "roberta-base"
MAX_LENGTH = 512

# Email / confidence thresholds
EMAIL_TEXT_MIN_LEN = 80
EMAIL_SCORE_MIN = 0.4
PHISHING_STRONG = 0.8
LEGIT_STRONG = 0.8
GRAY_LOW = 0.4
GRAY_HIGH = 0.6


def get_model_dir() -> Path:
    return DEFAULT_MODEL_DIR


def score_email_likeness(text: str) -> float:
    """
    Heuristic score in [0,1] for how much this text looks like an email.
    Uses simple cues: length, headers, URLs, and email addresses.
    """
    if not (text or "").strip():
        return 0.0

    t = (text or "").lower()
    score = 0.0

    length = len(t)
    if length >= 80:
        score += 0.2
    if length >= 200:
        score += 0.2
    if length >= 400:
        score += 0.1

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if any(ln.startswith("from:") for ln in lines):
        score += 0.2
    if any(ln.startswith("to:") for ln in lines):
        score += 0.1
    if any(ln.startswith("subject:") for ln in lines):
        score += 0.2
    if any("unsubscribe" in ln for ln in lines):
        score += 0.1
    if any("regards" in ln or "sincerely" in ln for ln in lines):
        score += 0.1

    if "http://" in t or "https://" in t or ".com" in t or ".net" in t or ".org" in t:
        score += 0.2

    if re.search(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+", t):
        score += 0.2

    return max(0.0, min(score, 1.0))


class PhishingPredictor:
    def __init__(self, model_dir: Path | str | None = None):
        model_dir = Path(model_dir or get_model_dir())
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}. Run training first: python -m src.train --model_name distilbert-base-uncased --max_samples 5000 --epochs 1"
            )
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), num_labels=2)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Run the transformer and return raw probabilities and argmax label.
        Keys: label, confidence, phishing_probability, legitimate_probability, text_preview.
        """
        text_norm = (text or "").strip()
        if not text_norm:
            return {
                "label": "legitimate",
                "confidence": 0.0,
                "phishing_probability": 0.0,
                "legitimate_probability": 1.0,
                "text_preview": "",
            }

        inputs = self.tokenizer(
            text_norm,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        p_legit = float(probs[0].item())
        p_phish = float(probs[1].item())
        pred_id = int(probs.argmax().item())
        confidence = max(p_legit, p_phish)
        preview = text_norm[:200] + "..." if len(text_norm) > 200 else text_norm

        return {
            "label": LABEL_NAMES[pred_id],
            "confidence": round(confidence, 4),
            "phishing_probability": round(p_phish, 4),
            "legitimate_probability": round(p_legit, 4),
            "text_preview": preview,
        }


_predictor: PhishingPredictor | None = None


def get_predictor() -> PhishingPredictor:
    global _predictor
    if _predictor is None:
        _predictor = PhishingPredictor()
    return _predictor


def classify_email_text(text: str) -> dict:
    """
    Apply email-likeness rules + phishing model to produce a richer classification.
    Returns keys matching PredictResponse in main.py.
    """
    text_norm = (text or "").strip()
    text_len = len(text_norm)
    email_score = score_email_likeness(text_norm)

    if not text_norm:
        return {
            "final_label": "not_email",
            "label": "not_email",
            "confidence": 0.0,
            "phishing_probability": 0.0,
            "legitimate_probability": 0.0,
            "email_score": round(email_score, 4),
            "reason": "no_text",
            "text_preview": "",
        }

    predictor = get_predictor()
    raw = predictor.predict(text_norm)
    p_phish = float(raw.get("phishing_probability", 0.0))
    p_legit = float(raw.get("legitimate_probability", 0.0))
    confidence = float(raw.get("confidence", max(p_phish, p_legit)))

    # Default final label is based on probabilities; may be overridden by rules below.
    final_label = raw.get("label", "legitimate")
    reason: str | None = None

    if text_len < EMAIL_TEXT_MIN_LEN or email_score < EMAIL_SCORE_MIN:
        final_label = "not_email"
        reason = "low_email_score" if email_score < EMAIL_SCORE_MIN else "short_text"
    else:
        if p_phish >= PHISHING_STRONG and p_phish > p_legit:
            final_label = "phishing"
            reason = "strong_phishing"
        elif p_legit >= LEGIT_STRONG and p_legit > p_phish:
            final_label = "legitimate"
            reason = "strong_legitimate"
        elif GRAY_LOW <= confidence <= GRAY_HIGH:
            final_label = "uncertain"
            reason = "low_confidence_gray_zone"
        else:
            if p_phish > p_legit:
                final_label = "phishing"
                reason = "moderate_phishing"
            else:
                final_label = "legitimate"
                reason = "moderate_legitimate"

    return {
        "final_label": final_label,
        "label": final_label,
        "confidence": round(confidence, 4),
        "phishing_probability": round(p_phish, 4),
        "legitimate_probability": round(p_legit, 4),
        "email_score": round(email_score, 4),
        "reason": reason,
        "text_preview": raw.get("text_preview", ""),
    }
