"""
모델 평가 지표 계산.

- AUROC (per-class & macro-average)
- AUPRC (per-class & macro-average)
- Operating Point 최적화 (Youden's J)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        roc_curve,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

from src.train.models import DISEASE_LABELS


def compute_auroc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, float]:
    """
    클래스별 및 macro-average AUROC 계산.

    Args:
        y_true: (N, C) binary labels
        y_prob: (N, C) predicted probabilities
        labels: 클래스 이름 리스트

    Returns:
        {"Atelectasis": 0.82, ..., "macro_avg": 0.79}
    """
    assert _SKLEARN_AVAILABLE, "pip install scikit-learn 필요"

    result = {}
    for i, label in enumerate(labels):
        try:
            result[label] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            result[label] = float("nan")

    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro_avg"] = float(np.mean(valid)) if valid else float("nan")
    return result


def compute_auprc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, float]:
    """
    클래스별 및 macro-average AUPRC 계산.
    """
    assert _SKLEARN_AVAILABLE, "pip install scikit-learn 필요"

    result = {}
    for i, label in enumerate(labels):
        try:
            result[label] = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            result[label] = float("nan")

    valid = [v for v in result.values() if not np.isnan(v)]
    result["macro_avg"] = float(np.mean(valid)) if valid else float("nan")
    return result


def find_operating_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: List[str] = DISEASE_LABELS,
) -> Dict[str, float]:
    """
    각 클래스에 대해 Youden's J index 기준 최적 임계값 탐색.

    Returns:
        {"Atelectasis": 0.31, "Cardiomegaly": 0.42, ...}
    """
    assert _SKLEARN_AVAILABLE, "pip install scikit-learn 필요"

    thresholds = {}
    for i, label in enumerate(labels):
        try:
            fpr, tpr, thresh = roc_curve(y_true[:, i], y_prob[:, i])
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            thresholds[label] = float(thresh[best_idx])
        except Exception:
            thresholds[label] = 0.5
    return thresholds
