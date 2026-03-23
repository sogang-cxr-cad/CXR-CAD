# 🩺 CXR-CAD — Chest X-ray Computer-Aided Detection

> End-to-End Multi-label Classification CAD System for 14 Thoracic Diseases  
> Powered by DenseNet-121 · NIH ChestX-ray14 Dataset

---

## 🏗️ Architecture

**Microservices Architecture (MSA)**

```
┌─────────────┐     HTTP/JSON     ┌──────────────┐
│  Streamlit   │ ◄──────────────► │   FastAPI     │
│  Dashboard   │   localhost:8000  │   Backend     │
│  (port 8501) │                   │  (port 8000)  │
└─────────────┘                   └──────┬───────┘
                                         │
                                  ┌──────▼───────┐
                                  │  DenseNet-121 │
                                  │  PyTorch Model│
                                  └──────────────┘
```

## 📁 Project Structure

```text
CXR-CAD/
├── requirements.txt            # All dependencies
├── README.md                   # This file
├── src/                        # Core Logic
│   ├── preprocess/
│   │   ├── data_loader.py      # NIH Data Loader (placeholder)
│   │   └── transforms.py       # CLAHE, resizing, normalization
│   ├── train/
│   │   └── models.py           # DenseNet-121 architecture
│   └── analysis/               # Grad-CAM (to be implemented)
├── api/                        # Backend Service
│   ├── main.py                 # FastAPI app (/health, /predict)
│   └── schemas.py              # Pydantic schemas
└── dashboard/                  # Frontend Service
    └── app.py                  # Streamlit Dashboard
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend API

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger UI available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Start Frontend Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard opens at: [http://localhost:8501](http://localhost:8501)

### 4. Test the Pipeline

1. Open the Streamlit dashboard
2. Upload any chest X-ray image (PNG/JPEG)
3. View the AI analysis results with disease probabilities

> ⚠️ **Walking Skeleton Mode**: The current pipeline returns simulated predictions.  
> Real model inference will be available after training is complete.

## 🔬 Detected Diseases (14 Classes)

| # | Disease | # | Disease |
|---|---------|---|---------|
| 1 | Atelectasis | 8 | Pneumothorax |
| 2 | Cardiomegaly | 9 | Consolidation |
| 3 | Effusion | 10 | Edema |
| 4 | Infiltration | 11 | Emphysema |
| 5 | Mass | 12 | Fibrosis |
| 6 | Nodule | 13 | Pleural Thickening |
| 7 | Pneumonia | 14 | Hernia |

## 📋 Tech Stack

- **Model**: PyTorch · DenseNet-121 (ImageNet pretrained)
- **Backend**: FastAPI · Pydantic · Uvicorn
- **Frontend**: Streamlit · Plotly
- **Dataset**: NIH ChestX-ray14 (112,120 images)
