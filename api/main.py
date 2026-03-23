"""
CXR-CAD Backend API Service.

FastAPI application providing:
- GET  /health  → Service health check
- POST /predict → Accepts chest X-ray image, returns disease probabilities

Walking Skeleton: /predict returns simulated (fake) predictions for
end-to-end integration testing before real model inference is wired up.
"""

import time
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictionResult, HealthResponse


# ── Application Lifespan ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    print("🩺 CXR-CAD API starting up...")
    print("   Model loading: SKIPPED (walking skeleton mode)")
    app.state.model_loaded = False  # Will be True when real model is loaded
    yield
    print("🩺 CXR-CAD API shutting down...")


# ── FastAPI App ────────────────────────────────────────────────────
app = FastAPI(
    title="CXR-CAD API",
    description=(
        "Chest X-ray Computer-Aided Detection API.\n\n"
        "Multi-label classification for 14 thoracic diseases using DenseNet-121 "
        "trained on NIH ChestX-ray14 dataset.\n\n"
        "**Walking Skeleton Mode** — returning simulated predictions."
    ),
    version="0.1.0-skeleton",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Disease Labels ─────────────────────────────────────────────────
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# Realistic fake probabilities for the walking skeleton
_FAKE_PROBS = {
    "Atelectasis": 0.32,
    "Cardiomegaly": 0.85,
    "Effusion": 0.50,
    "Infiltration": 0.18,
    "Mass": 0.12,
    "Nodule": 0.08,
    "Pneumonia": 0.22,
    "Pneumothorax": 0.05,
    "Consolidation": 0.15,
    "Edema": 0.42,
    "Emphysema": 0.03,
    "Fibrosis": 0.07,
    "Pleural_Thickening": 0.11,
    "Hernia": 0.02,
}

# Placeholder Grad-CAM Base64 (1×1 red pixel PNG)
_FAKE_GRADCAM_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8BQDwAEgAF/"
    "poIuwwAAAABJRU5ErkJggg=="
)

DETECTION_THRESHOLD = 0.3


# ── Endpoints ──────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    tags=["System"],
)
async def health_check():
    """Return service health status and model readiness."""
    return HealthResponse(
        status="healthy",
        model_loaded=app.state.model_loaded,
        version=app.version,
    )


@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Predict Diseases from Chest X-ray",
    tags=["Inference"],
)
async def predict(file: UploadFile = File(..., description="Chest X-ray image (PNG/JPEG)")):
    """
    Accept a chest X-ray image and return multi-label disease probabilities.

    **Walking Skeleton Mode:** Returns simulated predictions with small random
    noise to demonstrate end-to-end integration flow.
    """
    # Validate file type
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a PNG or JPEG image.",
        )

    # Read file to ensure it's a valid upload (even though we don't process it yet)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Simulate Inference ──────────────────────────────────────
    start_time = time.time()
    time.sleep(0.3)  # Simulated inference delay

    # Add small random noise to fake probabilities for realism
    probs = {
        disease: round(min(1.0, max(0.0, base_prob + random.uniform(-0.05, 0.05))), 4)
        for disease, base_prob in _FAKE_PROBS.items()
    }

    inference_ms = int((time.time() - start_time) * 1000)

    # ── Build Response ──────────────────────────────────────────
    detected = [d for d, p in probs.items() if p >= DETECTION_THRESHOLD]
    top_disease = max(probs, key=probs.get)

    return PredictionResult(
        **probs,
        Detected_Diseases=detected,
        Top_Disease=top_disease,
        GradCAM_Base64=_FAKE_GRADCAM_B64,
        Inference_Time_ms=inference_ms,
    )
