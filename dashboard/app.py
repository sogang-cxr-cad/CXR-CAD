"""
CXR-CAD — Professional Medical Imaging Dashboard.

Streamlit-based frontend for the Chest X-ray CAD system.
Communicates with the FastAPI backend at localhost:8000 via HTTP.
"""

import requests
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
import io

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="CXR-CAD | Chest X-ray AI Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

DETECTION_THRESHOLD = 0.3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom CSS for Premium Medical UI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
    /* ── Import Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ─────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* ── Premium Card ───────────────────────────────────────────── */
    .premium-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }

    /* ── Top Disease Card ───────────────────────────────────────── */
    .top-disease-card {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(220, 38, 38, 0.35);
        margin-bottom: 1rem;
    }
    .top-disease-card h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: white !important;
    }
    .top-disease-card .prob {
        font-size: 3rem;
        font-weight: 300;
        margin: 0.25rem 0;
        color: #fecaca !important;
    }
    .top-disease-card .label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.8;
        color: #fecaca !important;
    }

    /* ── Detected Disease Tag ───────────────────────────────────── */
    .disease-tag {
        display: inline-block;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        color: #991b1b;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
        border: 1px solid #fecaca;
    }

    /* ── Status Badge ───────────────────────────────────────────── */
    .status-connected {
        background: #065f46;
        color: #a7f3d0;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-disconnected {
        background: #7f1d1d;
        color: #fecaca;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* ── Inference Metric ───────────────────────────────────────── */
    .inference-metric {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .inference-metric .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1e40af;
    }
    .inference-metric .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #3b82f6;
    }

    /* ── Header ─────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: white !important;
    }
    .main-header p {
        margin: 0.25rem 0 0 0;
        font-size: 0.85rem;
        opacity: 0.7;
        color: #94a3b8 !important;
    }

    /* ── Section Title ──────────────────────────────────────────── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* ── Hide Streamlit branding ────────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def check_api_health() -> bool:
    """Check if the backend API is reachable."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def call_predict_api(image_bytes: bytes, filename: str) -> dict | None:
    """Send image to backend /predict endpoint."""
    try:
        # Determine content type
        ext = filename.lower().split(".")[-1]
        content_type = "image/png" if ext == "png" else "image/jpeg"

        files = {"file": (filename, image_bytes, content_type)}
        resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)

        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error {resp.status_code}: {resp.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API. Is the server running?")
        return None


def get_risk_color(prob: float) -> str:
    """Return color hex based on probability risk level."""
    if prob >= 0.5:
        return "#dc2626"   # Red — High risk
    elif prob >= 0.3:
        return "#f59e0b"   # Amber — Medium risk
    else:
        return "#10b981"   # Green — Low risk


def create_disease_chart(probs: dict) -> go.Figure:
    """Create a horizontal bar chart of disease probabilities."""
    # Sort by probability (ascending for horizontal bars — highest on top)
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=False)
    diseases = [item[0].replace("_", " ") for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [get_risk_color(v) for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=diseases,
        x=values,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=6,
        ),
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        textfont=dict(size=11, family="Inter", color="#475569"),
        hovertemplate="<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>",
    ))

    # Threshold line
    fig.add_vline(
        x=DETECTION_THRESHOLD,
        line=dict(color="#f59e0b", width=2, dash="dash"),
        annotation=dict(
            text=f"Threshold ({DETECTION_THRESHOLD:.0%})",
            font=dict(size=10, color="#f59e0b"),
            yref="paper",
            y=1.05,
        ),
    )

    fig.update_layout(
        height=480,
        margin=dict(l=0, r=40, t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 1.05],
            showgrid=True,
            gridcolor="#f1f5f9",
            tickformat=".0%",
            tickfont=dict(size=10, family="Inter", color="#94a3b8"),
        ),
        yaxis=dict(
            tickfont=dict(size=12, family="Inter", color="#334155", weight=500),
        ),
        font=dict(family="Inter"),
    )

    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🩺 CXR-CAD")
    st.markdown("*AI-Powered Chest X-ray Analysis*")
    st.markdown("---")

    # API status check
    api_connected = check_api_health()
    if api_connected:
        st.markdown('<span class="status-connected">● API Connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-disconnected">● API Disconnected</span>', unsafe_allow_html=True)

    st.markdown("---")

    # Image uploader
    st.markdown("### 📤 Upload X-ray Image")
    uploaded_file = st.file_uploader(
        "Select a chest X-ray image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        help="Upload a frontal (PA/AP) chest X-ray in PNG or JPEG format.",
    )

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Detection Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Minimum probability to flag a disease as detected.",
    )

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; opacity:0.5; font-size:0.75rem;'>"
        "CXR-CAD v0.1.0<br>Walking Skeleton<br>For Research Use Only"
        "</div>",
        unsafe_allow_html=True,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<div class="main-header">
    <div>
        <h1>🩺 CXR-CAD — Chest X-ray AI Diagnosis</h1>
        <p>Multi-label thoracic disease detection powered by DenseNet-121 · NIH ChestX-ray14</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Content
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if uploaded_file is None:
    # Empty state
    st.markdown("""
    <div class="premium-card" style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🫁</div>
        <h2 style="color: #1e293b; margin-bottom: 0.5rem;">Upload a Chest X-ray to Begin</h2>
        <p style="color: #64748b; font-size: 1rem; max-width: 500px; margin: 0 auto;">
            Use the sidebar to upload a frontal chest X-ray image (PA/AP view).
            The AI will analyze it for 14 thoracic diseases in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    feat_cols = st.columns(3)
    features = [
        ("🔬", "14 Disease Detection", "Simultaneous screening for atelectasis, cardiomegaly, effusion, and 11 more pathologies."),
        ("⚡", "Real-time Inference", "DenseNet-121 backbone delivers results in under 500ms with high sensitivity."),
        ("🎯", "Grad-CAM Visualization", "See exactly where the AI focuses attention for transparent, explainable diagnostics."),
    ]
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"""
            <div class="premium-card" style="text-align:center; min-height:180px;">
                <div style="font-size:2.5rem; margin-bottom:0.5rem;">{icon}</div>
                <h4 style="color:#1e293b; margin:0 0 0.5rem 0;">{title}</h4>
                <p style="color:#64748b; font-size:0.85rem; margin:0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Image uploaded — run analysis ───────────────────────────
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))

    col_left, col_right = st.columns([2, 3], gap="large")

    # ── Left Column: Image Display ──────────────────────────────
    with col_left:
        st.markdown('<div class="section-title">📷 Uploaded Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption=uploaded_file.name)
        st.markdown('</div>', unsafe_allow_html=True)

        # Grad-CAM placeholder
        st.markdown('<div class="section-title">🎯 Grad-CAM Overlay</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="premium-card" style="text-align:center; padding: 2rem;">
            <div style="font-size: 2rem; opacity:0.3;">🔬</div>
            <p style="color:#94a3b8; font-size:0.85rem; margin:0.5rem 0 0 0;">
                Grad-CAM visualization will appear here<br>after model training is complete.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Right Column: Analysis Results ──────────────────────────
    with col_right:
        st.markdown('<div class="section-title">📊 Analysis Results</div>', unsafe_allow_html=True)

        if not api_connected:
            st.error("⚠️ Backend API is not connected. Please start the FastAPI server first.")
            st.code("uvicorn api.main:app --reload --port 8000", language="bash")
        else:
            # Call the API
            with st.spinner("🔄 Analyzing chest X-ray..."):
                result = call_predict_api(image_bytes, uploaded_file.name)

            if result:
                # ── Top Disease Metric Card ─────────────────────
                top_disease = result["Top_Disease"].replace("_", " ")
                top_prob = result[result["Top_Disease"]]

                st.markdown(f"""
                <div class="top-disease-card">
                    <div class="label">Primary Finding</div>
                    <h2>{top_disease}</h2>
                    <div class="prob">{top_prob:.1%}</div>
                    <div class="label">Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Metrics Row ─────────────────────────────────
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="inference-metric">
                        <div class="value">{result['Inference_Time_ms']}ms</div>
                        <div class="label">Inference Time</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    detected_count = len(result["Detected_Diseases"])
                    st.markdown(f"""
                    <div class="inference-metric">
                        <div class="value">{detected_count}</div>
                        <div class="label">Diseases Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="inference-metric">
                        <div class="value">14</div>
                        <div class="label">Diseases Screened</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Detected Diseases Tags ──────────────────────
                st.markdown('<div class="section-title">🚨 Detected Diseases</div>', unsafe_allow_html=True)
                detected = [d for d in DISEASE_LABELS if result.get(d, 0) >= threshold]

                if detected:
                    tags_html = " ".join(
                        f'<span class="disease-tag">{d.replace("_", " ")} ({result[d]:.0%})</span>'
                        for d in detected
                    )
                    st.markdown(f'<div class="premium-card">{tags_html}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="premium-card" style="text-align:center; color:#10b981;">
                        ✅ No significant pathology detected above threshold.
                    </div>
                    """, unsafe_allow_html=True)

                # ── Disease Probability Chart ───────────────────
                st.markdown('<div class="section-title">📈 All Disease Probabilities</div>', unsafe_allow_html=True)
                probs = {label: result[label] for label in DISEASE_LABELS}
                fig = create_disease_chart(probs)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
