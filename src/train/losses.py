"""
손실 함수 모듈.

구현 내용:
- FocalLoss: gamma 파라미터 (0, 1, 2) 실험 지원, pos_weight 적용
- gamma=0 일 때 가중치 적용 BCE와 동일하게 동작함을 보장
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-label Binary Focal Loss.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection" (2017)
        https://arxiv.org/abs/1708.02002

    수식:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (0 = weighted BCE, 1, 2 권장).
               gamma=0 → 가중 BCE와 동일.
        pos_weight: 양성 클래스 가중치 텐서 shape (num_classes,).
                    None이면 균등 가중치 사용.
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model outputs (before sigmoid), shape (B, C)
            targets: Binary labels, shape (B, C), values ∈ {0, 1}

        Returns:
            Scalar loss value
        """
        # BCE with logits (수치 안정성 보장)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",  # 각 원소별 손실 계산 후 focal weighting
        )

        # p_t 계산: 정답 클래스의 예측 확률
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Focal Loss = focal_weight * BCE
        focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

    def __repr__(self) -> str:
        return (
            f"FocalLoss(gamma={self.gamma}, "
            f"pos_weight={'set' if self.pos_weight is not None else 'None'}, "
            f"reduction='{self.reduction}')"
        )


def build_loss(
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> FocalLoss:
    """
    Loss 팩토리 함수.

    Args:
        gamma: Focal Loss gamma (0=BCE, 1, 2)
        pos_weight: 클래스별 양성 가중치 (data_loader.compute_pos_weight 결과)

    Returns:
        FocalLoss 인스턴스
    """
    loss_fn = FocalLoss(gamma=gamma, pos_weight=pos_weight)
    print(f"✅ Loss 설정: {loss_fn}")
    return loss_fn
