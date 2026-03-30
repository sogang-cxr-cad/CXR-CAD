"""
CXR-CAD Backend API Service.

FastAPI 애플리케이션:
- GET  /health          → 서비스 상태 확인
- GET  /models          → 지원 모델 목록
- POST /predict         → 흉부 X-ray 분석 (모델 선택 지원)

모델 선택: ?model=densenet | efficientnet | vit
체크포인트가 없으면 Placeholder 모드로 동작.
"""

from __future__ import annotations

import os
import time
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch

from api.schemas import HealthResponse, ModelInfoResponse, PredictionResult
from src.train.models import (
    DISEASE_LABELS,
    SUPPORTED_MODELS,
    build_model,
    get_model_info,
    load_checkpoint,
)
from src.preprocess.dicom_utils import is_dicom, dicom_to_pil
from src.preprocess.transforms import preprocess_single_image

# ── 설정 ─────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR    = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
DETECTION_THRESHOLD = 0.3
API_VERSION       = "0.2.0"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Placeholder Grad-CAM (1×1 빨간 픽셀 PNG)
_FAKE_GRADCAM_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8BQDwAEgAF/"
    "poIuwwAAAABJRU5ErkJggg=="
)

# ── 모델 레지스트리 ───────────────────────────────────────────────────────────

_model_registry: Dict[str, Optional[object]] = {k: None for k in SUPPORTED_MODELS}


def _find_checkpoint(model_key: str) -> Optional[Path]:
    """Best checkpoint 파일 자동 탐색 (가장 최근 fold 우선)."""
    if not CHECKPOINT_DIR.exists():
        return None
    candidates = sorted(CHECKPOINT_DIR.glob(f"{model_key}_fold*_best.pt"), reverse=True)
    return candidates[0] if candidates else None


def _load_model_if_available(model_key: str) -> bool:
    """체크포인트가 있으면 모델 로드. 없으면 None 유지."""
    ckpt = _find_checkpoint(model_key)
    if ckpt is None:
        return False
    try:
        model = build_model(model_key, pretrained=False)
        model = load_checkpoint(model, str(ckpt), device=str(DEVICE))
        model.eval()
        _model_registry[model_key] = model
        print(f"  ✅ {model_key}: {ckpt.name}")
        return True
    except Exception as e:
        print(f"  ⚠️  {model_key} 로드 실패: {e}")
        return False


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 사용 가능한 모든 모델 자동 로드."""
    print(f"🩺 CXR-CAD API v{API_VERSION} 시작 중...")
    print(f"   Device: {DEVICE}")

    loaded_any = False
    for key in SUPPORTED_MODELS:
        if _load_model_if_available(key):
            loaded_any = True

    if not loaded_any:
        print("   ℹ️  체크포인트 없음 → Placeholder 모드로 실행")

    app.state.loaded_models = [k for k, v in _model_registry.items() if v is not None]
    yield
    print("🩺 CXR-CAD API 종료")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="CXR-CAD API",
    description=(
        "흉부 X-ray 컴퓨터 보조 진단 API.\n\n"
        "**지원 모델**: DenseNet-121, EfficientNet-B4, ViT-B/16\n\n"
        "`?model=densenet|efficientnet|vit` 파라미터로 모델 선택.\n\n"
        "체크포인트 학습 전에는 Placeholder 모드로 동작합니다."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 추론 헬퍼 ─────────────────────────────────────────────────────────────────

def _simulate_prediction(model_key: str) -> Dict[str, float]:
    """실제 모델 없을 때 사용하는 Placeholder 예측값 (모델별로 다른 패턴)."""
    base = {
        "densenet": {
            "Atelectasis": 0.32, "Cardiomegaly": 0.85, "Effusion": 0.50,
            "Infiltration": 0.18, "Mass": 0.12, "Nodule": 0.08,
            "Pneumonia": 0.22, "Pneumothorax": 0.05, "Consolidation": 0.15,
            "Edema": 0.42, "Emphysema": 0.03, "Fibrosis": 0.07,
            "Pleural_Thickening": 0.11, "Hernia": 0.02,
        },
        "efficientnet": {
            "Atelectasis": 0.29, "Cardiomegaly": 0.88, "Effusion": 0.53,
            "Infiltration": 0.20, "Mass": 0.14, "Nodule": 0.09,
            "Pneumonia": 0.24, "Pneumothorax": 0.06, "Consolidation": 0.17,
            "Edema": 0.45, "Emphysema": 0.04, "Fibrosis": 0.08,
            "Pleural_Thickening": 0.12, "Hernia": 0.02,
        },
        "vit": {
            "Atelectasis": 0.31, "Cardiomegaly": 0.83, "Effusion": 0.48,
            "Infiltration": 0.22, "Mass": 0.11, "Nodule": 0.07,
            "Pneumonia": 0.21, "Pneumothorax": 0.04, "Consolidation": 0.14,
            "Edema": 0.40, "Emphysema": 0.03, "Fibrosis": 0.06,
            "Pleural_Thickening": 0.10, "Hernia": 0.01,
        },
    }
    base_probs = base.get(model_key, base["densenet"])
    return {
        d: round(min(1.0, max(0.0, p + random.uniform(-0.04, 0.04))), 4)
        for d, p in base_probs.items()
    }


def _run_real_inference(model_key: str, image: Image.Image) -> Dict[str, float]:
    """실제 모델로 추론."""
    model = _model_registry[model_key]
    tensor = preprocess_single_image(image).to(DEVICE)
    with torch.no_grad():
        probs = model(tensor).squeeze(0).cpu().tolist()
    return {disease: round(float(p), 4) for disease, p in zip(DISEASE_LABELS, probs)}


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """서비스 상태 및 모델 로드 여부 반환."""
    loaded = app.state.loaded_models
    return HealthResponse(
        status="healthy",
        model_loaded=len(loaded) > 0,
        loaded_models=loaded,
        version=API_VERSION,
    )


@app.get("/models", response_model=ModelInfoResponse, tags=["System"])
async def list_models():
    """지원 모델 목록 및 정보 반환 (UI 표시용)."""
    info = get_model_info()
    for key in SUPPORTED_MODELS:
        info[key]["is_loaded"] = _model_registry[key] is not None
    return ModelInfoResponse(models=info)


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="흉부 X-ray 이미지 (PNG/JPEG) 또는 DICOM (.dcm)"),
    model: str = Query(
        default="densenet",
        description="사용할 모델: densenet | efficientnet | vit",
    ),
    threshold: float = Query(
        default=DETECTION_THRESHOLD,
        ge=0.0, le=1.0,
        description="질환 감지 임계값",
    ),
):
    """
    흉부 X-ray를 업로드하고 14개 질환 확률을 반환합니다.

    - **model**: `densenet` (DenseNet-121), `efficientnet` (EfficientNet-B4), `vit` (ViT-B/16)
    - **threshold**: 이 값 이상의 확률을 감지된 질환으로 분류
    - DICOM 파일도 지원됩니다 (.dcm 확장자)
    """
    # 모델 유효성 검사
    model_key = model.lower().strip()
    if model_key not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 모델: '{model}'. 지원 목록: {SUPPORTED_MODELS}",
        )

    # 파일 읽기
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="빈 파일이 업로드되었습니다.")

    # 이미지 변환 (DICOM 또는 일반 이미지)
    filename = file.filename or ""
    try:
        if filename.lower().endswith((".dcm", ".dicom")) or is_dicom(io.BytesIO(contents)):
            # DICOM → PIL
            import tempfile, os as _os
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            try:
                image = dicom_to_pil(tmp_path)
            finally:
                _os.unlink(tmp_path)
        else:
            # 일반 이미지 (PNG/JPEG)
            allowed_types = ("image/png", "image/jpeg", "image/jpg")
            if file.content_type and file.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"지원하지 않는 파일 형식: {file.content_type}. PNG/JPEG/DICOM을 사용하세요.",
                )
            image = Image.open(io.BytesIO(contents)).convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 오류: {str(e)}")

    # 추론
    start_time = time.time()

    if _model_registry[model_key] is not None:
        probs = _run_real_inference(model_key, image)
    else:
        # Placeholder 모드 (체크포인트 없을 때)
        time.sleep(0.3)
        probs = _simulate_prediction(model_key)

    inference_ms = int((time.time() - start_time) * 1000)

    # 결과 집계
    detected    = [d for d, p in probs.items() if p >= threshold]
    top_disease = max(probs, key=probs.get)

    # 모델 표시명 가져오기
    model_display_name = get_model_info().get(model_key, {}).get("display_name", model_key)

    return PredictionResult(
        **probs,
        Detected_Diseases    = detected,
        Top_Disease          = top_disease,
        GradCAM_Base64       = _FAKE_GRADCAM_B64,  # 실제 모델 있을 때 Grad-CAM으로 교체
        Inference_Time_ms    = inference_ms,
        Model_Used           = model_display_name,
        Model_Key            = model_key,
    )
