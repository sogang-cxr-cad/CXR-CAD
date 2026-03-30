"""
Focal Loss 유닛 테스트.

- gamma=0 시 가중 BCE와 수치 동일 여부
- gamma 증가 시 loss 감소 여부 (easy sample weighting)
- pos_weight 적용 여부
"""

import torch
import pytest
from src.train.losses import FocalLoss, build_loss


class TestFocalLoss:
    def setup_method(self):
        torch.manual_seed(42)
        self.B, self.C = 4, 14
        self.logits  = torch.randn(self.B, self.C)
        self.targets = torch.randint(0, 2, (self.B, self.C)).float()

    def test_focal_gamma0_equals_bce(self):
        """gamma=0 이면 FocalLoss == BCEWithLogitsLoss (pos_weight 없을 때)."""
        fl   = FocalLoss(gamma=0.0, pos_weight=None)
        bce  = torch.nn.BCEWithLogitsLoss()
        fl_val  = fl(self.logits, self.targets)
        bce_val = bce(self.logits, self.targets)
        assert abs(fl_val.item() - bce_val.item()) < 1e-5, (
            f"gamma=0 FocalLoss({fl_val:.6f}) ≠ BCELoss({bce_val:.6f})"
        )

    def test_focal_higher_gamma_lower_easy_loss(self):
        """쉬운 샘플(확률 1.0에 가까운)일수록 gamma 증가 시 loss 감소."""
        # 확률이 높은 쉬운 배치 생성
        easy_logits  = torch.full((self.B, self.C), 5.0)   # sigmoid → ~0.99
        easy_targets = torch.ones(self.B, self.C)

        loss_g0 = FocalLoss(gamma=0)(easy_logits, easy_targets)
        loss_g2 = FocalLoss(gamma=2)(easy_logits, easy_targets)
        assert loss_g2 < loss_g0, (
            f"gamma=2 loss({loss_g2:.6f}) should be < gamma=0 loss({loss_g0:.6f})"
        )

    def test_pos_weight_increases_loss_for_positives(self):
        """pos_weight가 높을수록 양성 클래스에 대한 loss 증가."""
        pw_low  = torch.ones(self.C)
        pw_high = torch.full((self.C,), 10.0)

        all_pos_logits  = torch.zeros(self.B, self.C)
        all_pos_targets = torch.ones(self.B, self.C)

        loss_low  = FocalLoss(gamma=2, pos_weight=pw_low)(all_pos_logits, all_pos_targets)
        loss_high = FocalLoss(gamma=2, pos_weight=pw_high)(all_pos_logits, all_pos_targets)
        assert loss_high > loss_low, "pos_weight=10 loss should be > pos_weight=1 loss"

    def test_output_shape_reduction_none(self):
        """reduction='none' 시 출력 shape = 입력 shape."""
        fl  = FocalLoss(gamma=2, reduction="none")
        out = fl(self.logits, self.targets)
        assert out.shape == self.logits.shape

    def test_build_loss_factory(self):
        """build_loss 팩토리 함수 정상 반환."""
        pw = torch.ones(self.C)
        fl = build_loss(gamma=1.0, pos_weight=pw)
        assert isinstance(fl, FocalLoss)
        assert fl.gamma == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
