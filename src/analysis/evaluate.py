"""
CXR-CAD 모델 평가 모듈.

구현 내용:
- AUROC / AUPRC 계산
- Youden's J 기반 최적 임계값 결정
- Operating Point 분석 (Sens@Spec90, Spec@Sens90)
- Calibration Curve, ECE (Expected Calibration Error)
- Temperature Scaling (ECE > 0.10 자동 적용)
- Subgroup Analysis (성별 / 연령대 / View Position)
- External Validation (CheXpert, PadChest)
- Domain Shift 분석
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from src.preprocess.data_loader import DISEASE_LABELS, NUM_CLASSES


# ── AUROC / AUPRC ─────────────────────────────────────────────────────────────

def compute_auroc_auprc(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    클래스별 AUROC 및 AUPRC 계산.

    Args:
        labels: (N, 14) binary labels
        probs : (N, 14) predicted probabilities

    Returns:
        {
            "auroc": {"Atelectasis": 0.82, ...},
            "auprc": {"Atelectasis": 0.45, ...},
            "auroc_mean": 0.81,
            "auprc_mean": 0.42,
        }
    """
    auroc_per_class = {}
    auprc_per_class = {}

    for i, disease in enumerate(DISEASE_LABELS):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            auroc_per_class[disease] = float("nan")
            auprc_per_class[disease] = float("nan")
            continue

        try:
            auroc_per_class[disease] = float(roc_auc_score(y_true, y_prob))
            auprc_per_class[disease] = float(average_precision_score(y_true, y_prob))
        except Exception:
            auroc_per_class[disease] = float("nan")
            auprc_per_class[disease] = float("nan")

    valid_aurocs = [v for v in auroc_per_class.values() if not np.isnan(v)]
    valid_auprcs = [v for v in auprc_per_class.values() if not np.isnan(v)]

    return {
        "auroc":      auroc_per_class,
        "auprc":      auprc_per_class,
        "auroc_mean": float(np.mean(valid_aurocs)) if valid_aurocs else float("nan"),
        "auprc_mean": float(np.mean(valid_auprcs)) if valid_auprcs else float("nan"),
    }


# ── Youden's J 최적 임계값 ────────────────────────────────────────────────────

def find_optimal_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Dict[str, float]:
    """
    Youden's J Statistic으로 클래스별 최적 임계값 결정.
    J = Sensitivity + Specificity - 1 = TPR - FPR (최대화 지점)

    Returns:
        {disease_name: optimal_threshold}
    """
    thresholds = {}
    for i, disease in enumerate(DISEASE_LABELS):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            thresholds[disease] = 0.5
            continue

        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        thresholds[disease] = float(thresh[best_idx])

    return thresholds


# ── Operating Point 분석 ──────────────────────────────────────────────────────

def compute_operating_points(
    labels: np.ndarray,
    probs: np.ndarray,
    target_spec: float = 0.90,
    target_sens: float = 0.90,
) -> Dict[str, Dict]:
    """
    임상적 의사결정 지원을 위한 Operating Point 분석.

    - Sens@Spec90: 특이도 90% 조건에서의 민감도 (스크리닝 용도)
    - Spec@Sens90: 민감도 90% 조건에서의 특이도 (확진 용도)

    Returns:
        {
            disease: {
                "sens_at_spec90": float,
                "spec_at_sens90": float,
                "screening_threshold": float,  # Spec@Sens90을 위한 임계값
                "confirmation_threshold": float,
            }
        }
    """
    results = {}
    for i, disease in enumerate(DISEASE_LABELS):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            results[disease] = {
                "sens_at_spec90": float("nan"),
                "spec_at_sens90": float("nan"),
                "screening_threshold":    float("nan"),
                "confirmation_threshold": float("nan"),
            }
            continue

        fpr, tpr, thresh = roc_curve(y_true, y_prob)
        specificity = 1.0 - fpr

        # Sens@Spec90: spec >= 0.90 구간에서 최대 sens
        mask_spec90 = specificity >= target_spec
        if mask_spec90.any():
            sens_at_spec90 = float(tpr[mask_spec90].max())
            conf_thresh_idx = np.where(mask_spec90)[0][np.argmax(tpr[mask_spec90])]
            confirmation_threshold = float(thresh[min(conf_thresh_idx, len(thresh)-1)])
        else:
            sens_at_spec90 = float("nan")
            confirmation_threshold = float("nan")

        # Spec@Sens90: sens >= 0.90 구간에서 최대 spec
        mask_sens90 = tpr >= target_sens
        if mask_sens90.any():
            spec_at_sens90 = float(specificity[mask_sens90].max())
            screen_thresh_idx = np.where(mask_sens90)[0][np.argmax(specificity[mask_sens90])]
            screening_threshold = float(thresh[min(screen_thresh_idx, len(thresh)-1)])
        else:
            spec_at_sens90 = float("nan")
            screening_threshold = float("nan")

        results[disease] = {
            "sens_at_spec90":         sens_at_spec90,
            "spec_at_sens90":         spec_at_sens90,
            "screening_threshold":    screening_threshold,      # 스크리닝 용도 (높은 민감도)
            "confirmation_threshold": confirmation_threshold,  # 확진 용도 (높은 특이도)
        }

    return results


# ── ECE & Calibration ─────────────────────────────────────────────────────────

def compute_ece(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """
    Expected Calibration Error (ECE) 계산.

    Args:
        labels: (N, 14) binary labels
        probs : (N, 14) predicted probabilities
        n_bins: 보정 구간 수

    Returns:
        {"ece_mean": float, "ece_per_class": {disease: float}}
    """
    ece_per_class = {}
    for i, disease in enumerate(DISEASE_LABELS):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            ece_per_class[disease] = float("nan")
            continue

        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        n = len(y_true)
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            acc = y_true[mask].mean()
            conf = y_prob[mask].mean()
            ece += (mask.sum() / n) * abs(acc - conf)

        ece_per_class[disease] = float(ece)

    valid_eces = [v for v in ece_per_class.values() if not np.isnan(v)]
    return {
        "ece_mean":      float(np.mean(valid_eces)) if valid_eces else float("nan"),
        "ece_per_class": ece_per_class,
    }


# ── Temperature Scaling ───────────────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Temperature Scaling — Soft Calibration.

    logits = logits / temperature  →  ECE 최소화 방향으로 T 최적화.
    ECE > 0.10인 경우 적용 권장.

    Reference:
        Guo et al., "On Calibration of Modern Neural Networks" (2017)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        model: nn.Module,
        val_loader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        검증 세트에서 temperature 파라미터 최적화.

        Returns:
            최적화된 temperature 값
        """
        model.eval()
        self.to(device)

        all_logits = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                probs = model(images)
                logits = torch.log(probs.clamp(1e-7, 1 - 1e-7) / (1 - probs.clamp(1e-7, 1 - 1e-7)))
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self.forward(all_logits), all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        temp_val = float(self.temperature.item())
        print(f"✅ Temperature Scaling 완료: T = {temp_val:.4f}")
        return temp_val


# ── Subgroup Analysis ─────────────────────────────────────────────────────────

def subgroup_analysis(
    labels: np.ndarray,
    probs: np.ndarray,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    성별 / 연령대 / View Position 서브그룹 분석.

    Args:
        labels  : (N, 14) binary labels
        probs   : (N, 14) predicted probabilities
        meta_df : 메타데이터 DataFrame (patient_sex, patient_age, view_position 컬럼 필요)

    Returns:
        DataFrame with columns: [subgroup, subgroup_value, disease, auroc, n_samples]
    """
    records = []

    subgroup_cols = {
        "patient_sex":      lambda df: df["patient_sex"].fillna("Unknown"),
        "age_group":        _assign_age_group,
        "view_position":    lambda df: df["view_position"].fillna("Unknown"),
    }

    for sg_name, sg_func in subgroup_cols.items():
        groups = sg_func(meta_df)
        unique_groups = groups.unique()

        for group_val in unique_groups:
            mask = (groups == group_val).values
            if mask.sum() < 10:  # 너무 적은 샘플 제외
                continue

            sub_labels = labels[mask]
            sub_probs  = probs[mask]

            for i, disease in enumerate(DISEASE_LABELS):
                y_true = sub_labels[:, i]
                y_prob = sub_probs[:, i]

                if y_true.sum() < 2:
                    continue

                try:
                    auroc = float(roc_auc_score(y_true, y_prob))
                except Exception:
                    auroc = float("nan")

                records.append({
                    "subgroup":       sg_name,
                    "subgroup_value": str(group_val),
                    "disease":        disease,
                    "auroc":          auroc,
                    "n_samples":      int(mask.sum()),
                })

    return pd.DataFrame(records)


def _assign_age_group(meta_df: pd.DataFrame) -> pd.Series:
    """연령을 10세 단위 그룹으로 분류."""
    ages = pd.to_numeric(meta_df["patient_age"], errors="coerce").fillna(-1)
    bins   = [-1, 0, 20, 40, 60, 80, float("inf")]
    labels = ["Unknown", "0-20", "21-40", "41-60", "61-80", "80+"]
    return pd.cut(ages, bins=bins, labels=labels, right=True)


# ── External Validation ───────────────────────────────────────────────────────

DOMAIN_SHIFT_CAUSES = [
    "스캐너 모델 및 X-ray 장비 차이 (Manufacturer, Acquisition Protocol)",
    "환자 인구통계 분포 차이 (연령/성별/인종 분포)",
    "레이블링 방식 차이 (NLP 추출 vs 방사선과 전문의 직접 레이블)",
    "View Position 분포 차이 (PA/AP 비율)",
    "이미지 해상도 및 전처리 파이프라인 차이",
]


def summarize_domain_shift(
    internal_auroc: Dict[str, float],
    external_auroc: Dict[str, float],
    dataset_name: str = "CheXpert",
) -> pd.DataFrame:
    """
    내부(NIH) vs 외부 데이터셋(CheXpert 등) AUROC 비교 및 Domain Shift 분석.

    Returns:
        DataFrame: disease, nih_auroc, external_auroc, delta 컬럼
    """
    records = []
    for disease in DISEASE_LABELS:
        nih_auc = internal_auroc.get(disease, float("nan"))
        ext_auc = external_auroc.get(disease, float("nan"))
        delta   = ext_auc - nih_auc if not (np.isnan(nih_auc) or np.isnan(ext_auc)) else float("nan")
        records.append({
            "disease":        disease,
            "nih_auroc":      nih_auc,
            f"{dataset_name}_auroc": ext_auc,
            "delta":          delta,
        })

    df = pd.DataFrame(records)

    print(f"\n📊 Domain Shift 분석 — NIH vs {dataset_name}")
    print(df.to_string(index=False, float_format="{:.4f}".format))
    print(f"\n🔍 Domain Shift 주요 원인 (3가지 이상):")
    for j, cause in enumerate(DOMAIN_SHIFT_CAUSES[:3], 1):
        print(f"  {j}. {cause}")

    return df
