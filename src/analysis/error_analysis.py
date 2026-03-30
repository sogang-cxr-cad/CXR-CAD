"""
에러 분석 모듈.

구현 내용:
- False Positive (FP) 분석: 5건 이상
- False Negative (FN) 분석: 5건 이상
- 폐 영역 이탈 케이스 분석 (Grad-CAM 기반)
- Shortcut Learning 판정
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocess.data_loader import DISEASE_LABELS


# ── FP / FN 케이스 추출 ───────────────────────────────────────────────────────

def extract_fp_fn_cases(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: Dict[str, float],
    meta_df: Optional[pd.DataFrame] = None,
    top_n: int = 10,
) -> Dict[str, pd.DataFrame]:
    """
    클래스별 FP/FN 케이스 추출.

    Args:
        labels:     (N, 14) binary labels
        probs:      (N, 14) predicted probabilities
        thresholds: Youden's J 기반 최적 임계값 딕셔너리
        meta_df:    메타데이터 DataFrame (image_index, patient_id 등)
        top_n:      추출할 케이스 수 (기본 10)

    Returns:
        {
            "fp": DataFrame (FP 케이스, 클래스별 top_n개),
            "fn": DataFrame (FN 케이스, 클래스별 top_n개),
        }
    """
    fp_records = []
    fn_records = []

    for i, disease in enumerate(DISEASE_LABELS):
        threshold = thresholds.get(disease, 0.5)
        y_true = labels[:, i]
        y_prob = probs[:, i]
        y_pred = (y_prob >= threshold).astype(int)

        # FP: 실제 음성 (0) → 예측 양성 (1)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fp_indices = np.where(fp_mask)[0]
        fp_probs_sorted = np.argsort(y_prob[fp_indices])[::-1][:top_n]
        for idx in fp_indices[fp_probs_sorted]:
            record = {
                "disease":   disease,
                "case_type": "FP",
                "sample_idx": int(idx),
                "true_label": int(y_true[idx]),
                "pred_prob":  float(y_prob[idx]),
                "threshold":  float(threshold),
            }
            if meta_df is not None and idx < len(meta_df):
                row = meta_df.iloc[idx]
                record.update({
                    "image_index":   row.get("image_index", ""),
                    "patient_age":   row.get("patient_age", ""),
                    "patient_sex":   row.get("patient_sex", ""),
                    "view_position": row.get("view_position", ""),
                })
            fp_records.append(record)

        # FN: 실제 양성 (1) → 예측 음성 (0)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_indices = np.where(fn_mask)[0]
        fn_probs_sorted = np.argsort(y_prob[fn_indices])[:top_n]  # 가장 낮은 확률 순
        for idx in fn_indices[fn_probs_sorted]:
            record = {
                "disease":    disease,
                "case_type":  "FN",
                "sample_idx": int(idx),
                "true_label": int(y_true[idx]),
                "pred_prob":  float(y_prob[idx]),
                "threshold":  float(threshold),
            }
            if meta_df is not None and idx < len(meta_df):
                row = meta_df.iloc[idx]
                record.update({
                    "image_index":   row.get("image_index", ""),
                    "patient_age":   row.get("patient_age", ""),
                    "patient_sex":   row.get("patient_sex", ""),
                    "view_position": row.get("view_position", ""),
                })
            fn_records.append(record)

    fp_df = pd.DataFrame(fp_records)
    fn_df = pd.DataFrame(fn_records)

    _print_summary("FP", fp_df, top_k=5)
    _print_summary("FN", fn_df, top_k=5)

    return {"fp": fp_df, "fn": fn_df}


def _print_summary(case_type: str, df: pd.DataFrame, top_k: int = 5):
    """FP/FN 요약 출력."""
    if df.empty:
        print(f"[{case_type}] 케이스 없음")
        return

    print(f"\n📋 {case_type} 케이스 분석 (상위 {top_k}개 / 클래스)")
    print(f"{'='*60}")
    count_per_disease = df.groupby("disease").size().sort_values(ascending=False)
    print(f"  클래스별 {case_type} 케이스 수:\n{count_per_disease.to_string()}")

    if "image_index" in df.columns:
        sample = df.head(top_k)[["disease", "image_index", "pred_prob", "patient_sex", "view_position"]]
        print(f"\n  {case_type} 샘플 {top_k}건:\n{sample.to_string(index=False)}")


# ── 폐 영역 이탈 케이스 분석 ──────────────────────────────────────────────────

def analyze_lung_deviation_cases(
    cam_results: List[Dict],
    min_cases: int = 5,
) -> pd.DataFrame:
    """
    Grad-CAM 분석 결과에서 폐 영역 이탈 케이스 분석.

    Args:
        cam_results: gradcam.detect_lung_deviation() 결과 딕셔너리 리스트
            각 항목: {"image_index": str, "disease": str, "lung_activation_ratio": float,
                       "is_deviated": bool, "peak_location": tuple}
        min_cases: 최소 이탈 케이스 수 (기본 5)

    Returns:
        폐 영역 이탈 케이스 DataFrame
    """
    df = pd.DataFrame(cam_results)
    if df.empty:
        return df

    deviated = df[df["is_deviated"] == True].copy()

    print(f"\n🫁 폐 영역 이탈 케이스 분석")
    print(f"  전체 케이스: {len(df)} / 이탈 케이스: {len(deviated)}")

    if len(deviated) >= min_cases:
        print(f"  ✅ 최소 {min_cases}건 이탈 케이스 확보")
        top_deviated = deviated.nsmallest(min(10, len(deviated)), "lung_activation_ratio")
        print(f"\n  이탈 케이스 상위 {len(top_deviated)}건:")
        cols = [c for c in ["image_index", "disease", "lung_activation_ratio", "peak_location"] if c in top_deviated.columns]
        print(top_deviated[cols].to_string(index=False))
    else:
        print(f"  ⚠️  최소 {min_cases}건 미달 (현재 {len(deviated)}건)")

    return deviated


# ── Shortcut Learning 판정 ───────────────────────────────────────────────────

def assess_shortcut_learning(
    cam_results: List[Dict],
    deviation_threshold: float = 0.3,
) -> Dict:
    """
    Shortcut Learning 판정.

    폐 영역 이탈 비율이 높으면 Shortcut Learning 의심.
    - 배경(병원 장비, 마킹 등) 영역에 집중 → Shortcut
    - 이탈 케이스 비율 > 30%이면 경고

    Args:
        cam_results: Grad-CAM 분석 결과 리스트
        deviation_threshold: Shortcut 판정 임계값 (기본 0.30 = 30%)

    Returns:
        {
            "shortcut_suspected": bool,
            "deviation_rate": float,
            "judgment": str,
            "recommendation": str,
        }
    """
    df = pd.DataFrame(cam_results)
    if df.empty:
        return {
            "shortcut_suspected": False,
            "deviation_rate":     0.0,
            "judgment":           "데이터 없음 — 판정 불가",
            "recommendation":     "Grad-CAM 분석 먼저 실행 필요",
        }

    deviation_rate = float(df["is_deviated"].mean())
    shortcut_suspected = deviation_rate > deviation_threshold

    if shortcut_suspected:
        judgment = (
            f"⚠️  Shortcut Learning 의심 (폐 이탈률 {deviation_rate:.1%} > {deviation_threshold:.0%})\n"
            f"   모델이 병변 대신 배경(장비, 마킹, 횡경막 등)에 집중하는 경향 있음."
        )
        recommendation = (
            "1. 데이터 증강 강화 (RandomErasing, CutOut으로 배경 가림)\n"
            "2. 폐 세그멘테이션 마스크 기반 ROI 크롭 전처리 적용\n"
            "3. 더 많은 데이터 또는 외부 데이터셋 추가 학습"
        )
    else:
        judgment = (
            f"✅ Shortcut Learning 없음 (폐 이탈률 {deviation_rate:.1%} ≤ {deviation_threshold:.0%})\n"
            f"   모델이 폐 영역에 집중하여 결정을 내리는 것으로 판단."
        )
        recommendation = "현재 판단 근거가 적절함. 주기적으로 재확인 권장."

    print(f"\n🔍 Shortcut Learning 판정")
    print(f"  {judgment}")
    print(f"  권장 조치: {recommendation}")

    return {
        "shortcut_suspected": shortcut_suspected,
        "deviation_rate":     deviation_rate,
        "judgment":           judgment,
        "recommendation":     recommendation,
    }
