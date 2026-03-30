"""
모델 Forward Pass 유닛 테스트.

- DenseNet-121, EfficientNet-B4, ViT-B/16 출력 shape 검증
- 출력 범위 [0, 1] 검증 (sigmoid 포함 여부)
- 빌드 팩토리 / 정보 딕셔너리 검증
"""

import torch
import pytest
from src.train.models import (
    DenseNet121CAD, EfficientNetCAD, ViTCAD,
    SoftVotingEnsemble, build_model, get_model_info,
    SUPPORTED_MODELS, NUM_CLASSES,
)


FAKE_INPUT = torch.randn(2, 3, 224, 224)
EXPECTED_OUTPUT_SHAPE = (2, NUM_CLASSES)


class TestModels:

    @pytest.mark.parametrize("model_key", SUPPORTED_MODELS)
    def test_forward_output_shape(self, model_key):
        """모든 모델의 출력 shape이 (B, 14) 인지 확인."""
        model = build_model(model_key, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(FAKE_INPUT)
        assert out.shape == EXPECTED_OUTPUT_SHAPE, (
            f"{model_key}: 출력 shape {out.shape} ≠ {EXPECTED_OUTPUT_SHAPE}"
        )

    @pytest.mark.parametrize("model_key", SUPPORTED_MODELS)
    def test_output_range_is_probability(self, model_key):
        """출력값이 [0, 1] 범위 (sigmoid 적용 여부)."""
        model = build_model(model_key, pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(FAKE_INPUT)
        assert out.min().item() >= 0.0, f"{model_key}: 최솟값 < 0"
        assert out.max().item() <= 1.0, f"{model_key}: 최댓값 > 1"

    def test_soft_voting_ensemble_shape(self):
        """Soft Voting Ensemble 출력 shape 검증."""
        models_list = [
            build_model("densenet", pretrained=False),
            build_model("efficientnet", pretrained=False),
        ]
        ensemble = SoftVotingEnsemble(models_list)
        ensemble.eval()
        with torch.no_grad():
            out = ensemble(FAKE_INPUT)
        assert out.shape == EXPECTED_OUTPUT_SHAPE

    def test_soft_voting_is_average(self):
        """Soft Voting이 실제로 평균값을 반환하는지 확인."""
        m1 = build_model("densenet", pretrained=False)
        m1.eval()
        # 두 번째 모델을 첫 번째와 동일하게 설정
        m2 = build_model("densenet", pretrained=False)
        m2.load_state_dict(m1.state_dict())
        m2.eval()

        ensemble = SoftVotingEnsemble([m1, m2])
        ensemble.eval()
        with torch.no_grad():
            out_m1 = m1(FAKE_INPUT)
            out_e  = ensemble(FAKE_INPUT)
        assert torch.allclose(out_m1, out_e, atol=1e-5), "동일 모델 2개 앙상블 = 단일 모델과 동일해야 함"

    def test_build_model_invalid_key_raises(self):
        """지원하지 않는 모델명 입력 시 ValueError 발생."""
        with pytest.raises(ValueError):
            build_model("resnet50_invalid")

    def test_get_model_info_contains_all_keys(self):
        """get_model_info()가 모든 지원 모델 키를 포함하는지 확인."""
        info = get_model_info()
        for key in SUPPORTED_MODELS:
            assert key in info, f"'{key}' 모델 정보 없음"
            assert "display_name" in info[key]
            assert "params" in info[key]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
