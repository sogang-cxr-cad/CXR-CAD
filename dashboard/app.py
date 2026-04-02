"""
CXR-CAD — Professional Medical Imaging Dashboard.

Streamlit 기반 프론트엔드.
FastAPI 백엔드와 HTTP 통신.
DenseNet-121 / EfficientNet-B4 / ViT-B/16 모델 선택 UI 포함.
"""

from __future__ import annotations

import io
import os

import requests
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ── 설정 ─────────────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

MODEL_OPTIONS = {
    "densenet":    {"label": "🔗 DenseNet-121",    "params": "~8M",  "tag": "가볍고 빠름"},
    "efficientnet": {"label": "⚡ EfficientNet-B4", "params": "~19M", "tag": "균형 최적화"},
    "vit":         {"label": "🧠 ViT-B/16",        "params": "~86M", "tag": "전역 문맥 학습"},
}

# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CXR-CAD | Chest X-ray AI Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] hr { border-color: #334155 !important; }

    /* ── 모델 카드 ── */
    .model-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.3rem 0;
        transition: border-color 0.2s;
    }
    .model-card.selected {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.3);
    }
    .model-card .model-name { font-weight: 600; font-size: 0.95rem; color: #e2e8f0; margin: 0; }
    .model-card .model-meta { font-size: 0.78rem; color: #94a3b8; margin: 0.2rem 0 0 0; }
    .model-tag {
        display: inline-block;
        background: #1d4ed8;
        color: #bfdbfe;
        border-radius: 4px;
        padding: 0.1rem 0.5rem;
        font-size: 0.72rem;
        margin-left: 0.4rem;
    }

    /* ── 프리미엄 카드 ── */
    .premium-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07);
        margin-bottom: 1rem;
    }

    /* ── 최고 질환 카드 ── */
    .top-disease-card {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(220,38,38,0.35);
        margin-bottom: 1rem;
    }
    .top-disease-card h2 { margin: 0; font-size: 1.8rem; font-weight: 700; color: white !important; }
    .top-disease-card .prob { font-size: 3rem; font-weight: 300; margin: 0.25rem 0; color: #fecaca !important; }
    .top-disease-card .sub  { font-size: 0.82rem; letter-spacing: 2px; opacity: 0.8; color: #fecaca !important; }

    /* ── 상태 뱃지 ── */
    .status-connected    { background:#065f46; color:#a7f3d0; padding:0.3rem 0.9rem; border-radius:999px; font-size:0.78rem; font-weight:600; display:inline-block; }
    .status-disconnected { background:#7f1d1d; color:#fecaca; padding:0.3rem 0.9rem; border-radius:999px; font-size:0.78rem; font-weight:600; display:inline-block; }
    .status-placeholder  { background:#1e3a5f; color:#93c5fd; padding:0.3rem 0.9rem; border-radius:999px; font-size:0.78rem; font-weight:600; display:inline-block; }

    /* ── 질환 태그 ── */
    .disease-tag { display:inline-block; background:linear-gradient(135deg,#fef2f2,#fee2e2); color:#991b1b; padding:0.4rem 1rem; border-radius:999px; font-size:0.85rem; font-weight:600; margin:0.2rem; border:1px solid #fecaca; }

    /* ── 추론 지표 ── */
    .inference-metric { background:linear-gradient(135deg,#eff6ff,#dbeafe); border:1px solid #bfdbfe; border-radius:12px; padding:1rem; text-align:center; }
    .inference-metric .value { font-size:1.5rem; font-weight:700; color:#1e40af; }
    .inference-metric .label { font-size:0.72rem; text-transform:uppercase; letter-spacing:1.5px; color:#3b82f6; }

    /* ── 헤더 ── */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin:0; font-size:1.6rem; font-weight:700; color:white !important; }
    .main-header p  { margin:0.25rem 0 0; font-size:0.85rem; opacity:0.7; color:#94a3b8 !important; }

    .section-title { font-size:1.05rem; font-weight:600; color:#1e293b; margin-bottom:0.75rem; padding-bottom:0.5rem; border-bottom:2px solid #e2e8f0; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def check_api_health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def get_model_info_from_api() -> dict:
    try:
        r = requests.get(f"{API_URL}/models", timeout=3)
        if r.status_code == 200:
            return r.json().get("models", {})
    except Exception:
        pass
    return {}


def call_predict_api(image_bytes: bytes, filename: str, model_key: str, threshold: float) -> dict | None:
    ext = filename.lower().split(".")[-1]
    content_type = "image/png" if ext == "png" else "image/jpeg"
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            params={"model": model_key, "threshold": threshold},
            files={"file": (filename, image_bytes, content_type)},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        st.error(f"API 오류 {resp.status_code}: {resp.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ 백엔드 API에 연결할 수 없습니다.")
        return None


def get_risk_color(prob: float, threshold: float) -> str:
    if prob >= 0.5:          return "#dc2626"  # 빨강 — 고위험
    elif prob >= threshold:  return "#f59e0b"  # 주황 — 중위험
    else:                    return "#10b981"  # 초록 — 저위험


def create_disease_chart(probs: dict, threshold: float) -> go.Figure:
    sorted_items = sorted(probs.items(), key=lambda x: x[1])
    diseases = [k.replace("_", " ") for k, _ in sorted_items]
    values   = [v for _, v in sorted_items]
    colors   = [get_risk_color(v, threshold) for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=diseases, x=values, orientation="h",
        marker=dict(color=colors, cornerradius=6),
        text=[f"{v:.1%}" for v in values], textposition="outside",
        textfont=dict(size=11, family="Inter", color="#475569"),
        hovertemplate="<b>%{y}</b><br>확률: %{x:.2%}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line=dict(color="#f59e0b", width=2, dash="dash"),
        annotation=dict(text=f"임계값 ({threshold:.0%})", font=dict(size=10, color="#f59e0b"), yref="paper", y=1.05))
    fig.update_layout(
        height=480, margin=dict(l=0, r=45, t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 1.12], showgrid=True, gridcolor="#f1f5f9", tickformat=".0%", tickfont=dict(size=10, family="Inter", color="#94a3b8")),
        yaxis=dict(tickfont=dict(size=12, family="Inter", color="#334155")),
        font=dict(family="Inter"),
    )
    return fig


# ── 사이드바 ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🩺 CXR-CAD")
    st.markdown("*AI-Powered Chest X-ray Analysis*")
    st.divider()

    # ── API 상태 ──────────────────────────────────────────────────────────────
    health = check_api_health()
    if health:
        loaded = health.get("loaded_models", [])
        model_ver = health.get("model_version", "")
        ver_tag   = f" · {model_ver}" if model_ver else ""
        if loaded:
            st.markdown(
                f'<span class="status-connected">● API Connected ({len(loaded)}개 모델 로드됨{ver_tag})</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="status-placeholder">● API Connected (Placeholder 모드{ver_tag})</span>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<span class="status-disconnected">● API Disconnected</span>', unsafe_allow_html=True)

    st.divider()

    # ── 모델 선택 ─────────────────────────────────────────────────────────────
    st.markdown("### 🤖 모델 선택")

    # API에서 모델 로드 상태 가져오기
    api_model_info = get_model_info_from_api() if health else {}

    model_labels = [v["label"] for v in MODEL_OPTIONS.values()]
    model_keys   = list(MODEL_OPTIONS.keys())

    selected_label = st.radio(
        "분석에 사용할 모델:",
        options=model_labels,
        label_visibility="collapsed",
        key="model_radio",
    )
    selected_model_key = model_keys[model_labels.index(selected_label)]
    model_opt = MODEL_OPTIONS[selected_model_key]

    # 선택된 모델 정보 카드
    is_loaded = api_model_info.get(selected_model_key, {}).get("is_loaded", False)
    loaded_badge = "✅ 로드됨" if is_loaded else "🔄 Placeholder"
    api_desc = api_model_info.get(selected_model_key, {}).get("description", "")
    st.markdown(
        f"""
        <div class="model-card selected">
            <p class="model-name">{model_opt['label']}
                <span class="model-tag">{model_opt['tag']}</span>
            </p>
            <p class="model-meta">파라미터: {model_opt['params']} &nbsp;|&nbsp; {loaded_badge}</p>
            {"<p class='model-meta'>" + api_desc + "</p>" if api_desc else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 이미지 업로드 ─────────────────────────────────────────────────────────
    st.markdown("### 📤 X-ray 업로드")
    uploaded_file = st.file_uploader(
        "흉부 X-ray 업로드",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="PA/AP 전면 흉부 X-ray (PNG 또는 JPEG)",
        key="xray_uploader",
    )

    st.divider()

    # ── 설정 ─────────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ 설정")
    threshold = st.slider(
        "감지 임계값", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
        help="이 확률 이상의 질환을 '감지됨'으로 분류합니다.",
        key="threshold_slider",
    )

    st.divider()
    st.page_link("pages/analysis_results.py", label="상세 분석 결과 보기", icon="📈")
    
    st.divider()
    st.markdown(
        "<div style='text-align:center;opacity:0.45;font-size:0.72rem;'>"
        "CXR-CAD v0.2.0<br>For Research Use Only</div>",
        unsafe_allow_html=True,
    )


# ── 메인 헤더 ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🩺 CXR-CAD — Chest X-ray AI Diagnosis</h1>
    <p>Multi-label thoracic disease detection · DenseNet-121 / EfficientNet-B4 / ViT-B/16 · NIH ChestX-ray14</p>
</div>
""", unsafe_allow_html=True)


# ── 메인 콘텐츠 ───────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="premium-card" style="text-align:center;padding:4rem 2rem;">
        <div style="font-size:4rem;margin-bottom:1rem;">🫁</div>
        <h2 style="color:#1e293b;margin-bottom:0.5rem;">Upload a Chest X-ray to Begin</h2>
        <p style="color:#64748b;font-size:1rem;max-width:500px;margin:0 auto;">
            사이드바에서 모델을 선택하고 흉부 X-ray 이미지를 업로드하세요.<br>
            AI가 14가지 흉부 질환을 즉시 분석합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    feat_cols = st.columns(3)
    features = [
        ("🔗", "DenseNet-121", "Dense connectivity 기반 경량 모델. 빠른 추론과 안정적 성능."),
        ("⚡", "EfficientNet-B4", "Compound Scaling으로 정확도/효율 균형 최적화."),
        ("🧠", "ViT-B/16", "Self-Attention으로 전역 영역 관계를 학습. 최고 표현력."),
    ]
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center;min-height:180px;">
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">{icon}</div>
                <h4 style="color:#1e293b;margin:0 0 0.5rem;">{title}</h4>
                <p style="color:#64748b;font-size:0.85rem;margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── 이미지 업로드 완료 → 분석 ─────────────────────────────────────────────
    image_bytes = uploaded_file.getvalue()
    image       = Image.open(io.BytesIO(image_bytes))

    col_left, col_right = st.columns([2, 3], gap="large")

    # ── 좌측: 이미지 ─────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-title">📷 업로드된 이미지</div>', unsafe_allow_html=True)
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.image(image, width="stretch", caption=uploaded_file.name)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">🎯 Grad-CAM 시각화</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="premium-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2rem;opacity:0.3;">🔬</div>
            <p style="color:#94a3b8;font-size:0.85rem;margin:0.5rem 0 0;">
                Grad-CAM 시각화는 모델 학습 완료 후 표시됩니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── 우측: 분석 결과 ──────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-title">📊 분석 결과</div>', unsafe_allow_html=True)

        if not health:
            st.error("⚠️ 백엔드 API가 연결되지 않았습니다. 서버를 먼저 실행하세요.")
            st.code("uvicorn api.main:app --reload --port 8000", language="bash")
        else:
            model_label = MODEL_OPTIONS[selected_model_key]["label"]
            with st.spinner(f"🔄 {model_label}로 분석 중..."):
                result = call_predict_api(image_bytes, uploaded_file.name, selected_model_key, threshold)

            if result:
                # ── 사용 모델 표시 ────────────────────────────────────────────
                model_used = result.get("Model_Used", selected_model_key)
                is_placeholder = not health.get("model_loaded", False)
                mode_tag = "⚠️ Placeholder 모드" if is_placeholder else f"✅ {model_used}"
                st.info(f"**분석 모델**: {mode_tag}", icon="🤖")

                # ── Top Disease 카드 ──────────────────────────────────────────
                top_disease = result["Top_Disease"].replace("_", " ")
                top_prob    = result.get("Top_Probability",
                              result.get(result["Top_Disease"], 0.0))  # 하위 호환
                st.markdown(f"""
                <div class="top-disease-card">
                    <div class="sub">PRIMARY FINDING</div>
                    <h2>{top_disease}</h2>
                    <div class="prob">{top_prob:.1%}</div>
                    <div class="sub">CONFIDENCE SCORE</div>
                </div>
                """, unsafe_allow_html=True)

                # ── 지표 행 ──────────────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                detected_count = len(result["Detected_Diseases"])
                with m1:
                    st.markdown(f"""<div class="inference-metric"><div class="value">{result['Inference_Time_ms']}ms</div><div class="label">추론 시간</div></div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""<div class="inference-metric"><div class="value">{detected_count}</div><div class="label">감지된 질환</div></div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""<div class="inference-metric"><div class="value">14</div><div class="label">검사 질환</div></div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── 감지된 질환 태그 ──────────────────────────────────────────
                st.markdown('<div class="section-title">🚨 감지된 질환</div>', unsafe_allow_html=True)
                detected = [d for d in DISEASE_LABELS if result.get(d, 0) >= threshold]
                if detected:
                    tags_html = " ".join(
                        f'<span class="disease-tag">{d.replace("_"," ")} ({result[d]:.0%})</span>'
                        for d in detected
                    )
                    st.markdown(f'<div class="premium-card">{tags_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="premium-card" style="text-align:center;color:#10b981;">✅ 임계값 이상의 유의미한 질환이 감지되지 않았습니다.</div>""", unsafe_allow_html=True)

                # ── 질환 확률 차트 ────────────────────────────────────────────
                st.markdown('<div class="section-title">📈 전체 질환 확률</div>', unsafe_allow_html=True)
                probs = {label: result[label] for label in DISEASE_LABELS}
                fig = create_disease_chart(probs, threshold)
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
