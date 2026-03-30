# 🩺 CXR-CAD — Chest X-ray Computer-Aided Detection

> End-to-End Multi-label Classification CAD System for 14 Thoracic Diseases  
> DenseNet-121 · EfficientNet-B4 · ViT-B/16 · NIH ChestX-ray14 Dataset

---

## 🏗️ Architecture

```
┌─────────────────┐     HTTP/JSON      ┌──────────────────────┐
│    Streamlit     │ ◄────────────────► │     FastAPI           │
│    Dashboard     │   localhost:8000   │     Backend           │
│    (port 8501)   │                    │  /health /models      │
│                  │  ?model=densenet   │  /predict             │
│  [Model Select]  │      efficientnet  └──────────┬───────────┘
│  DenseNet-121    │      vit                       │
│  EfficientNet-B4 │                    ┌──────────▼───────────┐
│  ViT-B/16        │                    │  DenseNet-121         │
└─────────────────┘                    │  EfficientNet-B4      │
                                        │  ViT-B/16             │
                                        │  (PyTorch + CUDA)     │
                                        └──────────────────────┘
```

## 📁 Project Structure

```text
CXR-CAD/
├── Dockerfile                      # CUDA 12.1 + PyTorch 2.2.0 GPU 환경
├── docker-compose.yml              # API + Dashboard 멀티 컨테이너
├── requirements.txt                # 전체 의존성
│
├── src/
│   ├── preprocess/
│   │   ├── data_loader.py          # NIH CSV 파싱, Patient-ID Split, 누수 검증, pos_weight
│   │   ├── transforms.py           # cv2 CLAHE, 학습/추론/TTA 파이프라인
│   │   └── dicom_utils.py          # pydicom 메타데이터 파싱, DICOM→PIL 변환
│   ├── train/
│   │   ├── models.py               # DenseNet-121 / EfficientNet-B4 / ViT-B/16 + Ensemble + TTA
│   │   ├── losses.py               # Focal Loss (gamma=0,1,2), pos_weight
│   │   └── train.py                # 5-Fold GroupKFold, EarlyStopping, Cosine Annealing
│   └── analysis/
│       ├── evaluate.py             # AUROC/AUPRC, Youden's J, ECE, Temperature Scaling, Subgroup/Domain Shift
│       ├── gradcam.py              # Grad-CAM (3개 모델 공용), 폐 영역 이탈 감지
│       └── error_analysis.py       # FP/FN 분석, Shortcut Learning 판정
│
├── api/
│   ├── main.py                     # /health, /models, /predict?model=...
│   └── schemas.py                  # Pydantic 스키마 (모델 선택 필드 포함)
│
├── dashboard/
│   └── app.py                      # Streamlit Dashboard (모델 선택 UI)
│
└── tests/
    ├── test_losses.py              # Focal Loss 유닛 테스트 (5개)
    ├── test_models.py              # 모델 forward pass 테스트 (6개)
    └── test_api.py                 # API 엔드포인트 테스트 (10개)
```

## 🚀 Quick Start

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. API 서버 시작

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. 대시보드 시작

```bash
streamlit run dashboard/app.py
```

Dashboard: [http://localhost:8501](http://localhost:8501)

### 4. Docker로 전체 실행 (GPU 필요)

```bash
docker-compose up --build
```

### 5. 테스트 실행

```bash
pytest tests/ -v
```

### 6. 모델 학습 (NIH 데이터셋 준비 후)

```bash
# gamma=2, DenseNet-121, 5-Fold GroupKFold
python -m src.train.train --data_root /path/to/nih --model densenet --gamma 2

# EfficientNet-B4
python -m src.train.train --data_root /path/to/nih --model efficientnet --gamma 2

# ViT-B/16
python -m src.train.train --data_root /path/to/nih --model vit --gamma 1
```

> ⚠️ **Placeholder 모드**: 체크포인트가 없으면 시뮬레이션 예측을 반환합니다.  
> 학습 완료 후 `checkpoints/` 폴더에 `.pt` 파일이 생성되면 자동으로 실제 추론으로 전환됩니다.

---

## 🧠 지원 모델

| 모델 | 파라미터 | 특징 |
|------|---------|------|
| **DenseNet-121** | ~8M | Dense connectivity, 가볍고 빠름 |
| **EfficientNet-B4** | ~19M | Compound scaling, 정확도/효율 균형 |
| **ViT-B/16** | ~86M | Self-Attention 기반 전역 문맥 학습 |

API 호출 시 `?model=densenet|efficientnet|vit` 파라미터로 모델 선택.  
대시보드에서는 사이드바 라디오 버튼으로 선택 가능.

---

## 🔬 탐지 질환 (14 Classes)

| # | Disease | # | Disease |
|---|---------|---|---------| 
| 1 | Atelectasis | 8 | Pneumothorax |
| 2 | Cardiomegaly | 9 | Consolidation |
| 3 | Effusion | 10 | Edema |
| 4 | Infiltration | 11 | Emphysema |
| 5 | Mass | 12 | Fibrosis |
| 6 | Nodule | 13 | Pleural Thickening |
| 7 | Pneumonia | 14 | Hernia |

---

## 📊 평가 지표 (학습 완료 후 채워질 항목)

### 5-Fold GroupKFold AUROC (Mean ± Std)

| 모델 | AUROC | gamma |
|------|-------|-------|
| DenseNet-121 | TBD | 2 |
| EfficientNet-B4 | TBD | 2 |
| ViT-B/16 | TBD | 1 |
| **Soft Voting Ensemble** | **TBD** | — |

### Calibration (ECE)

| 모델 | ECE (before) | ECE (after Temperature Scaling) |
|------|-------------|--------------------------------|
| DenseNet-121 | TBD | TBD |
| EfficientNet-B4 | TBD | TBD |
| ViT-B/16 | TBD | TBD |

### External Validation (CheXpert)

| 모델 | NIH AUROC | CheXpert AUROC | Δ (Domain Shift) |
|------|-----------|----------------|------------------|
| DenseNet-121 | TBD | TBD | TBD |

---

## 🔍 Grad-CAM 분석 (학습 완료 후 채워질 항목)

- 폐 영역 이탈 케이스: TBD건
- FP 분석: TBD건
- FN 분석: TBD건
- Shortcut Learning 판정: TBD

---

## 📋 Tech Stack

| 구분 | 기술 |
|------|------|
| **ML Framework** | PyTorch 2.2 · torchvision · timm |
| **모델** | DenseNet-121 · EfficientNet-B4 · ViT-B/16 |
| **전처리** | OpenCV (CLAHE) · pydicom · albumentations |
| **평가** | scikit-learn · scipy |
| **Backend** | FastAPI · Pydantic · Uvicorn |
| **Frontend** | Streamlit · Plotly |
| **인프라** | Docker · CUDA 12.1 |
| **데이터셋** | NIH ChestX-ray14 (112,120 images, 14 classes) |
| **테스트** | pytest |
