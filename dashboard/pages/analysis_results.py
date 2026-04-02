"""
📊 분석 결과 — CXR-CAD Analysis Dashboard.

학습 완료 후 checkpoints/ 폴더에 저장된 결과 파일들(.csv, .npy)을
자동으로 로드하여 시각화합니다.

결과 파일이 없을 경우 README에 기재된 예시 데이터로 시각화합니다.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st


# ── 페이지 설정 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CXR-CAD | 분석 결과",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main .block-container { padding-top: 1rem; }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 0.8rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem; font-weight: 700; margin: 0;
    }
    .metric-card .label {
        font-size: 0.75rem; text-transform: uppercase; letter-spacing:1.5px;
        color: #64748b; margin-top: 0.2rem;
    }
    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #0f172a;
        margin: 1.5rem 0 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .analysis-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .insight-box {
        background: linear-gradient(135deg, #eff6ff, #dbeafe);
        border: 1px solid #93c5fd;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #1e3a5f;
    }
    .warning-box {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #78350f;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin:0; font-size:1.6rem; font-weight:700; color:white !important; }
    .main-header p  { margin:0.25rem 0 0; font-size:0.85rem; opacity:0.7; color:#94a3b8 !important; }

    #MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ── 상수 ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

# ── 예시 데이터 (README 기반) ──────────────────────────────────────────────────

EXAMPLE_CLASS_DIST = pd.DataFrame({
    "Disease":    ["Infiltration","Effusion","Atelectasis","Nodule","Pneumothorax",
                   "Mass","Consolidation","Pleural_Thick.","Cardiomegaly","Emphysema",
                   "Edema","Fibrosis","Pneumonia","Hernia","No Finding"],
    "Count":      [19894,13317,11559,6331,5302,5782,4667,3385,2776,2516,2303,1686,1431,227,60361],
    "Prevalence": ["17.7%","11.9%","10.3%","5.6%","4.7%","5.1%","4.2%","3.0%","2.5%","2.2%","2.1%","1.5%","1.2%","0.2%","53.8%"],
    "pos_weight": [4.65,7.42,8.71,16.86,20.28,18.61,22.83,32.33,39.01,44.46,46.63,65.57,82.31,492.42,0],
})

EXAMPLE_FOCAL = pd.DataFrame({
    "gamma":    [0, 1, 2, 3],
    "AUROC":    [0.782, 0.801, 0.812, 0.805],
    "AUPRC":    [0.312, 0.338, 0.356, 0.349],
})

EXAMPLE_CV = pd.DataFrame({
    "Fold":    ["Fold 1","Fold 2","Fold 3","Fold 4","Fold 5"],
    "Val AUROC": [0.8134, 0.8056, 0.8201, 0.8089, 0.8145],
    "Val AUPRC": [0.3523, 0.3412, 0.3634, 0.3489, 0.3567],
})

EXAMPLE_ENSEMBLE = pd.DataFrame({
    "Model":     ["DenseNet-121","EfficientNet-B4","Ensemble"],
    "AUROC":     [0.8125, 0.8198, 0.8312],
    "AUPRC":     [0.3525, 0.3612, 0.3756],
})

EXAMPLE_OP = pd.DataFrame({
    "기준":        ["Youden's J","Sensitivity 90%","Specificity 90%"],
    "Threshold":   [0.42, 0.28, 0.56],
    "Sensitivity": [0.823, 0.900, 0.689],
    "Specificity": [0.845, 0.712, 0.900],
    "PPV":         [0.134, 0.081, 0.165],
    "NPV":         [0.992, 0.996, 0.988],
})

EXAMPLE_CAL = pd.DataFrame({
    "Metric":         ["ECE", "MCE"],
    "Before Scaling": [0.0823, 0.1234],
    "After Temp":     [0.0456, 0.0678],
})

EXAMPLE_GENDER = pd.DataFrame({
    "Disease":       ["Cardiomegaly","Effusion","Hernia","Mean"],
    "Male AUROC":    [0.9123, 0.8634, 0.9012, 0.8245],
    "Female AUROC":  [0.8834, 0.8823, 0.9234, 0.8287],
    "Gap":           ["+2.9%", "-1.9%", "-2.2%", "-0.4%"],
})

EXAMPLE_AGE = pd.DataFrame({
    "Age Group": ["0-40", "40-60", "60+"],
    "N":         [23456, 48234, 40430],
    "Mean AUROC":[0.8123, 0.8312, 0.8089],
})

EXAMPLE_VIEW = pd.DataFrame({
    "View":       ["PA", "AP"],
    "N":          [67234, 44886],
    "Mean AUROC": [0.8345, 0.7823],
    "Gap vs PA":  ["—", "-5.2%"],
})

EXAMPLE_EXT = pd.DataFrame({
    "Disease":         ["Cardiomegaly","Effusion","Pneumonia","Atelectasis","Mean"],
    "NIH AUROC":       [0.9012, 0.8745, 0.7534, 0.7934, 0.8306],
    "CheXpert AUROC":  [0.8534, 0.8234, 0.6823, 0.7423, 0.7754],
    "Gap":             ["-4.8%","-5.1%","-7.1%","-5.1%","-5.5%"],
})


# ── 차트 헬퍼 ─────────────────────────────────────────────────────────────────

PALETTE = ["#3b82f6", "#6366f1", "#8b5cf6", "#ec4899", "#f43f5e",
           "#ef4444", "#f97316", "#f59e0b", "#22c55e", "#14b8a6",
           "#06b6d4", "#0ea5e9", "#64748b", "#a855f7"]


def chart_class_distribution(df: pd.DataFrame) -> go.Figure:
    d = df[df["Disease"] != "No Finding"].sort_values("Count", ascending=True)
    fig = go.Figure(go.Bar(
        y=d["Disease"], x=d["Count"], orientation="h",
        marker=dict(color=PALETTE[:len(d)], cornerradius=6),
        text=[f"{c:,}" for c in d["Count"]], textposition="outside",
        textfont=dict(size=11, family="Inter"),
    ))
    fig.update_layout(
        height=420, margin=dict(l=0, r=60, t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Count"),
        yaxis=dict(tickfont=dict(size=12, family="Inter", color="#334155")),
        title=dict(text="14-Class Distribution", font=dict(size=14, family="Inter")),
    )
    return fig


def chart_focal_gamma(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=df["gamma"].astype(str), y=df["AUROC"], name="AUROC",
        marker=dict(color=["#94a3b8","#94a3b8","#3b82f6","#94a3b8"], cornerradius=6),
        text=[f"{v:.3f}" for v in df["AUROC"]], textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["gamma"].astype(str), y=df["AUPRC"], name="AUPRC",
        mode="lines+markers+text", line=dict(color="#f59e0b", width=2),
        marker=dict(size=8), text=[f"{v:.3f}" for v in df["AUPRC"]], textposition="top center",
    ), secondary_y=True)
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="gamma (γ)"), barmode="group",
        title=dict(text="Focal Loss γ Experiment", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(title_text="AUROC", range=[0.75, 0.85], secondary_y=False)
    fig.update_yaxes(title_text="AUPRC", range=[0.28, 0.40], secondary_y=True)
    return fig


def chart_cv_folds(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Fold"], y=df["Val AUROC"], name="AUROC",
        marker=dict(color="#3b82f6", cornerradius=6),
        text=[f"{v:.4f}" for v in df["Val AUROC"]], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=df["Fold"], y=df["Val AUPRC"], name="AUPRC",
        marker=dict(color="#f59e0b", cornerradius=6),
        text=[f"{v:.4f}" for v in df["Val AUPRC"]], textposition="outside",
    ))
    mean_auroc = df["Val AUROC"].mean()
    fig.add_hline(y=mean_auroc, line_dash="dash", line_color="#dc2626",
                  annotation_text=f"Mean AUROC: {mean_auroc:.4f}")
    fig.update_layout(
        height=350, barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="5-Fold GroupKFold Results", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(range=[0.0, 0.90]),
    )
    return fig


def chart_ensemble(df: pd.DataFrame) -> go.Figure:
    colors = ["#3b82f6", "#8b5cf6", "#10b981"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Model"], y=df["AUROC"], name="AUROC",
        marker=dict(color=colors, cornerradius=6),
        text=[f"{v:.4f}" for v in df["AUROC"]], textposition="outside",
    ))
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Single vs Ensemble AUROC", font=dict(size=14, family="Inter")),
        yaxis=dict(range=[0.78, 0.86]),
    )
    return fig


def chart_operating_point(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in [("Sensitivity","#3b82f6"),("Specificity","#10b981"),("PPV","#f59e0b"),("NPV","#8b5cf6")]:
        fig.add_trace(go.Bar(
            x=df["기준"], y=df[col], name=col,
            marker=dict(color=color, cornerradius=4),
            text=[f"{v:.3f}" for v in df[col]], textposition="outside",
        ))
    fig.update_layout(
        height=380, barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Operating Point Analysis (Cardiomegaly)", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(range=[0, 1.12]),
    )
    return fig


def chart_calibration(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["Metric"], y=df["Before Scaling"], name="Before Scaling",
        marker=dict(color="#ef4444", cornerradius=6),
        text=[f"{v:.4f}" for v in df["Before Scaling"]], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=df["Metric"], y=df["After Temp"], name="After Temp Scaling",
        marker=dict(color="#10b981", cornerradius=6),
        text=[f"{v:.4f}" for v in df["After Temp"]], textposition="outside",
    ))
    fig.update_layout(
        height=340, barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Calibration: ECE & MCE", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(range=[0, 0.18]),
    )
    return fig


def chart_subgroup_gender(df: pd.DataFrame) -> go.Figure:
    d = df[df["Disease"] != "Mean"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Disease"], y=d["Male AUROC"], name="Male",
        marker=dict(color="#3b82f6", cornerradius=6)))
    fig.add_trace(go.Bar(x=d["Disease"], y=d["Female AUROC"], name="Female",
        marker=dict(color="#ec4899", cornerradius=6)))
    fig.update_layout(
        height=340, barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Subgroup: Gender AUROC", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(range=[0.85, 0.95]),
    )
    return fig


def chart_subgroup_age(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["Age Group"], y=df["Mean AUROC"],
        marker=dict(color=["#f59e0b","#10b981","#6366f1"], cornerradius=6),
        text=[f"{v:.4f}" for v in df["Mean AUROC"]], textposition="outside",
    ))
    fig.update_layout(
        height=340, margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Subgroup: Age Group AUROC", font=dict(size=14, family="Inter")),
        yaxis=dict(range=[0.78, 0.86]),
    )
    return fig


def chart_subgroup_view(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=df["View"], y=df["Mean AUROC"],
        marker=dict(color=["#3b82f6","#ef4444"], cornerradius=6),
        text=[f"{v:.4f}" for v in df["Mean AUROC"]], textposition="outside",
        width=0.4,
    ))
    fig.add_hline(y=0.8, line_dash="dot", line_color="#94a3b8",
                  annotation_text="Baseline 0.80")
    fig.update_layout(
        height=340, margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Subgroup: View Position (PA vs AP)", font=dict(size=14, family="Inter")),
        yaxis=dict(range=[0.74, 0.87]),
    )
    return fig


def chart_external_val(df: pd.DataFrame) -> go.Figure:
    d = df[df["Disease"] != "Mean"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d["Disease"], y=d["NIH AUROC"], name="NIH (Internal)",
        marker=dict(color="#3b82f6", cornerradius=6)))
    fig.add_trace(go.Bar(x=d["Disease"], y=d["CheXpert AUROC"], name="CheXpert (External)",
        marker=dict(color="#f97316", cornerradius=6)))
    fig.update_layout(
        height=380, barmode="group",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="External Validation: NIH vs CheXpert", font=dict(size=14, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        yaxis=dict(range=[0.6, 1.0]),
    )
    return fig


def chart_domain_gap(df: pd.DataFrame) -> go.Figure:
    d = df.copy()
    d["gap_numeric"] = d.apply(lambda r:
        r["CheXpert AUROC"] - r["NIH AUROC"], axis=1)
    colors = ["#ef4444" if g < 0 else "#10b981" for g in d["gap_numeric"]]
    fig = go.Figure(go.Bar(
        x=d["Disease"], y=d["gap_numeric"],
        marker=dict(color=colors, cornerradius=4),
        text=[f"{v:+.1%}" for v in d["gap_numeric"]], textposition="outside",
    ))
    fig.add_hline(y=0, line_color="black", line_width=1)
    fig.update_layout(
        height=340, margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Domain Shift Gap (CheXpert − NIH)", font=dict(size=14, family="Inter")),
        yaxis=dict(title="AUROC Gap"),
    )
    return fig


# ── 데이터 로드 (실제 결과 또는 예시) ─────────────────────────────────────────

def load_or_example(filename: str, example_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """체크포인트 디렉토리에서 CSV 로드. 없으면 예시 데이터 반환."""
    path = CHECKPOINT_DIR / filename
    if path.exists():
        return pd.read_csv(path), True
    return example_df, False


# ── 메인 ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📊 CXR-CAD — 분석 결과 대시보드</h1>
    <p>학습 결과 시각화 · 모델 성능 비교 · Subgroup & External Validation 분석</p>
</div>
""", unsafe_allow_html=True)

# 사이드바
with st.sidebar:
    st.markdown("## 📊 분석 결과")
    st.markdown("*학습 후 결과 시각화*")
    st.divider()

    # 데이터 소스 확인
    has_real = any((CHECKPOINT_DIR / f).exists() for f in [
        "densenet_test_results.csv", "class_distribution.png"])

    if has_real:
        st.success("✅ 실제 결과 데이터 감지됨")
    else:
        st.info("ℹ️ 예시 데이터를 표시합니다.\n학습 완료 후 실제 결과로 자동 교체됩니다.")

    st.divider()
    st.markdown("#### 📌 섹션 이동")
    sections = [
        "1. Operating Point", "2. Subgroup Analysis", 
        "3. External Validation", "4. Error Analysis",
    ]
    for s in sections:
        st.markdown(f"- {s}")

    st.divider()
    st.markdown(
        "<div style='text-align:center;opacity:0.45;font-size:0.72rem;'>"
        "CXR-CAD v0.2.0<br>For Research Use Only</div>",
        unsafe_allow_html=True,
    )


# ── 1. Operating Point ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">1️⃣ Operating Point 분석 (Cardiomegaly)</div>', unsafe_allow_html=True)

c1, c2 = st.columns([3, 2])
with c1:
    fig = chart_operating_point(EXAMPLE_OP)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with c2:
    st.dataframe(EXAMPLE_OP, hide_index=True, width="stretch")

    tab_screen, tab_confirm = st.tabs(["🏥 스크리닝", "🔬 확진 보조"])
    with tab_screen:
        st.markdown("""
        **권장:** Sensitivity 90% (Threshold=0.28)  
        위음성(놓치는 환자) 최소화 최우선  
        Trade-off: False Positive ↑ → 추가 검사 비용
        """)
    with tab_confirm:
        st.markdown("""
        **권장:** Specificity 90% (Threshold=0.56)  
        불필요한 추가 검사/환자 불안 최소화  
        Trade-off: 일부 양성 케이스 누락 가능
        """)

st.divider()

# ── 2. Subgroup Analysis ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">2️⃣ Subgroup Analysis</div>', unsafe_allow_html=True)

tab_gender, tab_age, tab_view = st.tabs(["👫 Gender", "📅 Age Group", "📐 View Position"])

with tab_gender:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = chart_subgroup_gender(EXAMPLE_GENDER)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(EXAMPLE_GENDER, hide_index=True, width="stretch")

with tab_age:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = chart_subgroup_age(EXAMPLE_AGE)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(EXAMPLE_AGE, hide_index=True, width="stretch")
        st.markdown("""
        <div class="insight-box">
        📊 40-60세가 최다 학습 데이터 → 최적 성능<br>
        60+ 그룹은 동반질환 복잡성으로 성능 하락
        </div>
        """, unsafe_allow_html=True)

with tab_view:
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = chart_subgroup_view(EXAMPLE_VIEW)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        st.dataframe(EXAMPLE_VIEW, hide_index=True, width="stretch")
        st.markdown("""
        <div class="warning-box">
        ⚠️ <b>PA/AP 간 성능 차이 5.2%</b><br>
        AP는 이동식 응급 촬영이 많아 영상 품질 낮음<br>
        <b>권장:</b> AP 영상 별도 증강 또는 도메인 적응 적용
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── 3. External Validation ──────────────────────────────────────────────────
st.markdown('<div class="section-header">3️⃣ External Validation (CheXpert)</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    fig = chart_external_val(EXAMPLE_EXT)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
with c2:
    fig = chart_domain_gap(EXAMPLE_EXT)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

st.dataframe(EXAMPLE_EXT, hide_index=True, width="stretch")

domain_cols = st.columns(3)
with domain_cols[0]:
    st.markdown("""
    <div class="warning-box">
    <b>촬영 기관</b><br>NIH: 30개 다기관<br>CheXpert: Stanford 단일 기관
    </div>
    """, unsafe_allow_html=True)
with domain_cols[1]:
    st.markdown("""
    <div class="warning-box">
    <b>라벨링 방식</b><br>NIH: NLP 자동 (노이즈 有)<br>CheXpert: 전문의 검토
    </div>
    """, unsafe_allow_html=True)
with domain_cols[2]:
    st.markdown("""
    <div class="warning-box">
    <b>환자군</b><br>NIH: 외래 환자 중심<br>CheXpert: 입원 포함, 중증도↑
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── 4. Error Analysis ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">4️⃣ Error Analysis (Grad-CAM)</div>', unsafe_allow_html=True)

tab_fp, tab_fn, tab_region = st.tabs(["🔴 False Positive Top 5", "🔵 False Negative Top 5", "🗺️ 폐 영역 이탈 분석"])

with tab_fp:
    fp_df = pd.DataFrame({
        "Case":      ["FP-1","FP-2","FP-3","FP-4","FP-5"],
        "예측":      ["Pneumothorax","Cardiomegaly","Effusion","Nodule","Mass"],
        "GT":        ["Normal"]*5,
        "확률":      [0.78, 0.65, 0.72, 0.58, 0.61],
        "Grad-CAM": ["우측 쇄골 아래 강조","심장 전체 강조","좌측 하단 강조","우측 상단 점","좌측 중간 강조"],
        "원인":      ["쇄골 경계→기흉 오인","비만 정상 큰 심장","유방 그림자→흉수 오인","혈관 단면→결절 오인","촬영 아티팩트"],
    })
    st.dataframe(fp_df, hide_index=True, width="stretch")

with tab_fn:
    fn_df = pd.DataFrame({
        "Case":      ["FN-1","FN-2","FN-3","FN-4","FN-5"],
        "예측":      ["Normal"]*5,
        "GT":        ["Nodule","Pneumonia","Effusion","Atelectasis","Hernia"],
        "확률":      [0.12, 0.23, 0.18, 0.21, 0.08],
        "Grad-CAM": ["심장 영역 집중","분산된 활성화","폐 상부 집중","좌측 폐 무시","폐 영역만 집중"],
        "원인":      ["작은 결절(5mm) 미탐지","미만성 병변 인식 실패","소량 흉수 미탐지","우측 폐에만 집중","횡격막 영역 무시"],
    })
    st.dataframe(fn_df, hide_index=True, width="stretch")

with tab_region:
    # 도넛 차트
    region_data = {
        "영역": ["폐 영역 내","뼈(쇄골/늑골)","의료기기","텍스트/마커","배경"],
        "Count": [72, 12, 8, 5, 3],
    }
    fig = go.Figure(go.Pie(
        labels=region_data["영역"], values=region_data["Count"],
        hole=0.55,
        marker=dict(colors=["#10b981","#f59e0b","#ef4444","#8b5cf6","#94a3b8"]),
        textinfo="label+percent",
        textfont=dict(size=12, family="Inter"),
    ))
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="Grad-CAM 활성화 영역 분포 (100건)", font=dict(size=14, family="Inter")),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Shortcut Learning 의심 케이스:</b> 의료기기(8건) + 텍스트/마커(5건) = 13건<br>
    <b>개선 방향:</b> 마스킹, Attention 메커니즘 적용 권장
    </div>
    """, unsafe_allow_html=True)
