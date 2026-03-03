"""
FastAPI app: POST /predict/image (screenshot + OCR), POST /predict/text (raw or subject+body).
Serves the frontend at GET /.
"""
import io
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .predict import classify_email_text


def _decode_image_to_numpy(image_bytes: bytes):
    """
    Decode raw image bytes into a NumPy array.
    Tries Pillow first, then falls back to OpenCV for formats Pillow can't handle.
    """
    import numpy as np

    pil_error: Exception | None = None

    # Try Pillow (supports PNG/JPEG and, with plugins, more formats)
    try:
        from PIL import Image

        # Optional AVIF/HEIC support if plugin is installed
        try:
            import pillow_avif  # type: ignore  # noqa: F401
        except Exception:
            try:
                import pillow_avif_plugin  # type: ignore  # noqa: F401
            except Exception:
                pass
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except Exception:
            pass

        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        return np.array(img)
    except Exception as e:  # Pillow failed; remember error and try OpenCV
        pil_error = e

    # Fallback: OpenCV (good WebP and other formats support)
    try:
        import cv2

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        img_np = None

    if img_np is None:
        detail = "Unsupported or corrupt image format. Please upload PNG, JPEG, WEBP, or HEIC/HEIF (requires pillow-heif)."
        if pil_error is not None:
            detail += f" (Pillow error: {pil_error})"
        raise HTTPException(status_code=400, detail=detail)

    return img_np


def _run_ocr(image_bytes: bytes) -> str:
    import easyocr

    img_np = _decode_image_to_numpy(image_bytes)

    try:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        result = reader.readtext(img_np)
        return " ".join([t[1] for t in result]).strip() if result else ""
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OCR failed: {str(e)}")


app = FastAPI(title="Phishing Email Detector", description="Upload screenshot or paste email text")

# Pydantic models for request/response
class PredictTextBody(BaseModel):
    text: str | None = None
    subject: str | None = None
    body: str | None = None


class PredictResponse(BaseModel):
    final_label: str
    label: str
    confidence: float
    phishing_probability: float
    legitimate_probability: float
    email_score: float | None = None
    reason: str | None = None
    text_preview: str | None = None


@app.post("/predict/image", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image (e.g. PNG, JPEG)")
    contents = await file.read()
    text = _run_ocr(contents)
    out = classify_email_text(text)
    return PredictResponse(**out)


@app.post("/predict/text", response_model=PredictResponse)
async def predict_text(body: PredictTextBody):
    if body.text:
        text = body.text
    elif body.subject is not None or body.body is not None:
        text = (body.subject or "") + " " + (body.body or "")
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'subject' and/or 'body'")
    out = classify_email_text(text)
    return PredictResponse(**out)


# Frontend: serve static HTML/JS from templates and static
STATIC_DIR = Path(__file__).resolve().parent / "static"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index():
    index_html = TEMPLATES_DIR / "index.html"
    if index_html.exists():
        return index_html.read_text(encoding="utf-8")
    return "<html><body><h1>Phishing Detector API</h1><p>Use POST /predict/image or POST /predict/text. Serve index.html from templates/ for the UI.</p></body></html>"


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
