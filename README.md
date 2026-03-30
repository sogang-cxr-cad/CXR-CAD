# 🩺 CXR-CAD — Chest X-ray Computer-Aided Detection

> End-to-End Multi-label Classification CAD System for 14 Thoracic Diseases  
> DenseNet-121 · EfficientNet-B4 · ViT-B/16 · NIH ChestX-ray14 Dataset

---

## 🏗️ Architecture

```
┌─────────────────┐     HTTP/JSON      ┌──────────────────────┐
│    Streamlit     │ ◄────────────────► │     FastAPI           │
│    Dashboard     │   localhost:8000   │     Backend           │
│    (port 8501)   │                    │  GET  /health         │
│                  │  ?model=densenet   │  GET  /models         │
│  [Model Select]  │      efficientnet  │  POST /predict        │
│  DenseNet-121    │      vit           └──────────┬───────────┘
│  EfficientNet-B4 │                               │ .pth 자동 탐색
│  ViT-B/16        │                    ┌──────────▼───────────┐
└─────────────────┘                    │  checkpoints/         │
                                        │  densenet_best.pth    │
                                        │  efficientnet_best.pth│
                                        │  vit_best.pth         │
                                        │  (없으면 Placeholder) │
                                        └──────────────────────┘
```

## 📁 Project Structure

```text
CXR-CAD/
├── Dockerfile                          # CUDA 12.1 + PyTorch 2.2.0 GPU 환경
├── docker-compose.yml                  # API + Dashboard 멀티 컨테이너
├── requirements.txt                    # 전체 의존성
│
├── configs/
│   └── config.yaml                     # 학습 하이퍼파라미터 (모델·데이터·학습 설정)
│
├── src/
│   ├── preprocess/
│   │   ├── data_loader.py              # NIH CSV 파싱, Patient-ID Split, pos_weight 계산
│   │   ├── dataset.py                  # PyTorch Dataset 클래스 (NIH ChestX-ray14)
│   │   ├── split.py                    # GroupKFold 기반 Patient-level 데이터 분할
│   │   ├── transforms.py               # CLAHE, 학습/추론/TTA 변환 파이프라인
│   │   └── dicom_utils.py              # pydicom 메타데이터 파싱, DICOM→PIL 변환
│   │
│   ├── train/
│   │   ├── models.py                   # DenseNet-121 / EfficientNet-B4 / ViT-B/16 정의
│   │   ├── focal_loss.py               # Focal Loss (gamma=0,1,2) + pos_weight
│   │   ├── ensemble.py                 # Soft Voting Ensemble (3개 모델)
│   │   └── trainer.py                  # 5-Fold GroupKFold, EarlyStopping, Cosine Annealing
│   │
│   └── analysis/
│       ├── evaluation.py               # AUROC/AUPRC, F1, Confusion Matrix
│       ├── calibration.py              # ECE, Temperature Scaling
│       ├── gradcam.py                  # Grad-CAM (3개 모델 공용), 폐 영역 이탈 감지
│       ├── subgroup.py                 # 성별·연령대별 Subgroup 분석
│       └── external_val.py             # CheXpert 도메인 시프트 검증
│
├── api/
│   ├── main.py                         # /health, /models, /predict (DICOM 지원)
│   └── schemas.py                      # Pydantic 스키마 (요청·응답 모델)
│
├── dashboard/
│   └── app.py                          # Streamlit Dashboard (모델 선택 UI)
│
├── notebooks/
│   ├── 01_EDA.ipynb                    # 데이터 탐색 및 클래스 분포
│   ├── 02_CLAHE_Analysis.ipynb         # 전처리 효과 시각화
│   ├── 03_Focal_Loss_Experiment.ipynb  # gamma 파라미터 실험
│   ├── 04_Training.ipynb               # Colab 학습 실행 노트북
│   ├── 05_Operating_Point.ipynb        # Youden's J 임계값 최적화
│   ├── 06_Calibration.ipynb            # Temperature Scaling, ECE 측정
│   ├── 07_Subgroup_Analysis.ipynb      # 성별·연령 공정성 평가
│   ├── 08_External_Validation.ipynb    # CheXpert 외부 검증
│   └── 09_Error_Analysis.ipynb         # FP/FN, Shortcut Learning 분석
│
├── checkpoints/                        # ⚠️ .gitignore 처리 — .pth 파일 저장 위치
│   ├── densenet_best.pth               # (학습 후 배치)
│   ├── efficientnet_best.pth           # (학습 후 배치)
│   └── vit_best.pth                    # (학습 후 배치)
│
└── tests/
    ├── conftest.py                     # pytest fixtures
    ├── test_api.py                     # API 엔드포인트 통합 테스트
    ├── test_encoding.py                # 이미지 인코딩/디코딩 테스트
    └── test_transforms.py              # 전처리 변환 파이프라인 테스트
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

### 6. 모델 학습 (Colab 권장)

`notebooks/04_Training.ipynb`를 Google Colab에서 실행하거나, 아래 CLI를 사용합니다:

```bash
# configs/config.yaml 설정 후 실행
# DenseNet-121 (gamma=2, 5-Fold GroupKFold)
python -m src.train.trainer --config configs/config.yaml --model densenet

# EfficientNet-B4
python -m src.train.trainer --config configs/config.yaml --model efficientnet

# ViT-B/16
python -m src.train.trainer --config configs/config.yaml --model vit
```

학습이 완료되면 `checkpoints/<model_key>_best.pth` 형식으로 저장됩니다.

> ⚠️ **Placeholder 모드**: `checkpoints/`에 `.pth` 파일이 없으면 시뮬레이션 예측값을 반환합니다.  
> 체크포인트가 배치되면 서버 재시작 없이 자동으로 실제 추론으로 전환됩니다.

---

## 🧠 체크포인트 저장 포맷

Colab 학습 코드와 호환되는 표준 포맷:

```python
torch.save({
    "epoch"               : epoch,
    "model_state_dict"    : model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "val_auroc"           : best_auroc,
}, "checkpoints/<model_key>_best.pth")
```

API 서버는 `model_state_dict`, `state_dict`, 직접 state_dict 세 가지 포맷을 모두 지원합니다.

---

## 🧠 지원 모델

| 모델 | 파라미터 | 특징 |
|------|---------|------|
| **DenseNet-121** | ~8M | Dense connectivity, 가볍고 빠름 |
| **EfficientNet-B4** | ~19M | Compound scaling, 정확도/효율 균형 |
| **ViT-B/16** | ~86M | Self-Attention 기반 전역 문맥 학습 |
| **Soft Voting Ensemble** | — | 3개 모델 확률 평균 |

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
| **설정** | YAML (configs/config.yaml) |
