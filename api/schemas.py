"""
Pydantic Schemas for CXR-CAD API.

Defines request/response models for the prediction endpoint,
ensuring strict type validation and auto-documentation in Swagger UI.
"""

from typing import List

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    """
    Response schema for the /predict endpoint.

    Contains per-disease probabilities (0.0–1.0), summary fields,
    and Grad-CAM visualization data.
    """

    # ── Per-Disease Probabilities ──────────────────────────────────
    Atelectasis: float = Field(..., ge=0.0, le=1.0, description="Probability of Atelectasis")
    Cardiomegaly: float = Field(..., ge=0.0, le=1.0, description="Probability of Cardiomegaly")
    Effusion: float = Field(..., ge=0.0, le=1.0, description="Probability of Effusion")
    Infiltration: float = Field(..., ge=0.0, le=1.0, description="Probability of Infiltration")
    Mass: float = Field(..., ge=0.0, le=1.0, description="Probability of Mass")
    Nodule: float = Field(..., ge=0.0, le=1.0, description="Probability of Nodule")
    Pneumonia: float = Field(..., ge=0.0, le=1.0, description="Probability of Pneumonia")
    Pneumothorax: float = Field(..., ge=0.0, le=1.0, description="Probability of Pneumothorax")
    Consolidation: float = Field(..., ge=0.0, le=1.0, description="Probability of Consolidation")
    Edema: float = Field(..., ge=0.0, le=1.0, description="Probability of Edema")
    Emphysema: float = Field(..., ge=0.0, le=1.0, description="Probability of Emphysema")
    Fibrosis: float = Field(..., ge=0.0, le=1.0, description="Probability of Fibrosis")
    Pleural_Thickening: float = Field(..., ge=0.0, le=1.0, description="Probability of Pleural Thickening")
    Hernia: float = Field(..., ge=0.0, le=1.0, description="Probability of Hernia")

    # ── Summary Fields ─────────────────────────────────────────────
    Detected_Diseases: List[str] = Field(
        ..., description="List of diseases with probability above threshold"
    )
    Top_Disease: str = Field(..., description="Disease with the highest probability")
    GradCAM_Base64: str = Field(
        ..., description="Base64-encoded Grad-CAM heatmap overlay image"
    )
    Inference_Time_ms: int = Field(
        ..., ge=0, description="Model inference time in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                    "Detected_Diseases": ["Cardiomegaly", "Effusion", "Edema"],
                    "Top_Disease": "Cardiomegaly",
                    "GradCAM_Base64": "iVBORw0KGgoAAAANSUhEUg...",
                    "Inference_Time_ms": 312,
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded in memory")
    version: str = Field(..., description="API version string")
