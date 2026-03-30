"""
CXR-CAD 모델 정의.

지원 아키텍처:
- DenseNet121CAD  : DenseNet-121 (ImageNet pretrained)
- EfficientNetCAD : EfficientNet-B4 (ImageNet pretrained)
- ViTCAD          : Vision Transformer ViT-B/16 (ImageNet pretrained)
- SoftVotingEnsemble : 위 모델들의 Soft Voting 앙상블
- TTAWrapper         : Test-Time Augmentation 래퍼
"""

from __future__ import annotations

from typing import List, Optional, Dict

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ── 상수 ─────────────────────────────────────────────────────────────────────

DISEASE_LABELS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

NUM_CLASSES = len(DISEASE_LABELS)

# 지원 모델 이름 목록 (API, UI에서 사용)
SUPPORTED_MODELS = ["densenet", "efficientnet", "vit"]


# ── DenseNet-121 ──────────────────────────────────────────────────────────────

class DenseNet121CAD(nn.Module):
    """
    DenseNet-121 기반 Multi-label 흉부 X-ray 분류 모델.

    - Backbone: DenseNet-121 (ImageNet pretrained)
    - Head: Linear(1024 → 14) + Sigmoid
    - Input : (B, 3, 224, 224)
    - Output: (B, 14) probabilities ∈ [0, 1]

    특징:
    - Dense connectivity로 gradient vanishing 완화
    - 파라미터 효율적 (~8M)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Feature extractor (classifier 전까지)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = backbone.classifier.in_features  # 1024
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        out = torch.relu(features, inplace=True)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return torch.sigmoid(self.classifier(out))

    @property
    def model_name(self) -> str:
        return "DenseNet-121"

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── EfficientNet-B4 ───────────────────────────────────────────────────────────

class EfficientNetCAD(nn.Module):
    """
    EfficientNet-B4 기반 Multi-label 흉부 X-ray 분류 모델.

    - Backbone: EfficientNet-B4 (ImageNet pretrained)
    - Head: Linear(1792 → 14) + Sigmoid
    - Input : (B, 3, 224, 224)
    - Output: (B, 14) probabilities ∈ [0, 1]

    특징:
    - Compound scaling으로 정확도/효율 균형 최적화
    - DenseNet 대비 높은 표현력 (~19M)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b4(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        in_features = backbone.classifier[1].in_features  # 1792
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return torch.sigmoid(self.classifier(x))

    @property
    def model_name(self) -> str:
        return "EfficientNet-B4"

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Vision Transformer (ViT-B/16) ─────────────────────────────────────────────

class ViTCAD(nn.Module):
    """
    Vision Transformer ViT-B/16 기반 Multi-label 분류 모델.

    torchvision의 vit_b_16을 backbone으로 사용.
    timm이 설치되어 있을 경우 'vit_base_patch16_224'를 사용.

    - Input : (B, 3, 224, 224)
    - Output: (B, 14) probabilities ∈ [0, 1]

    특징:
    - Self-attention으로 전역 문맥 포착
    - 파라미터 수 많음 (~86M), 큰 데이터셋에서 강력
    """

    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()

        if _TIMM_AVAILABLE:
            # timm 사용 (더 유연한 pretrained 옵션)
            self.backbone = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0,  # head 제거
            )
            in_features = self.backbone.num_features  # 768
        else:
            # torchvision fallback
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.vit_b_16(weights=weights)
            # head를 Identity로 대체
            in_features = backbone.heads.head.in_features  # 768
            backbone.heads.head = nn.Identity()
            self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=0.1),
            nn.Linear(in_features, num_classes),
        )

        self._use_timm = _TIMM_AVAILABLE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return torch.sigmoid(self.classifier(features))

    @property
    def model_name(self) -> str:
        return "ViT-B/16"

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Soft Voting Ensemble ──────────────────────────────────────────────────────

class SoftVotingEnsemble(nn.Module):
    """
    여러 CAD 모델의 Soft Voting 앙상블.

    각 모델의 예측 확률 평균을 최종 예측으로 사용.
    가중치(weights)를 지정하면 가중 평균 사용.

    Args:
        models_list: CAD 모델 리스트
        weights: 각 모델의 가중치 (None이면 균등)
    """

    def __init__(
        self,
        models_list: List[nn.Module],
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(models_list)

        if weights is not None:
            assert len(weights) == len(models_list), "weights 길이가 models_list와 달라야 함"
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            self.weights = [1.0 / len(models_list)] * len(models_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_probs = torch.zeros(
            x.size(0), NUM_CLASSES, device=x.device, dtype=x.dtype
        )
        for model, weight in zip(self.models, self.weights):
            weighted_probs += weight * model(x)
        return weighted_probs

    @property
    def model_name(self) -> str:
        names = [getattr(m, "model_name", type(m).__name__) for m in self.models]
        return f"Ensemble({'+'.join(names)})"


# ── Test-Time Augmentation Wrapper ────────────────────────────────────────────

class TTAWrapper(nn.Module):
    """
    Test-Time Augmentation 래퍼.

    여러 augmentation view에 대해 forward를 실행하고 확률 평균을 반환.

    Args:
        model: 기반 CAD 모델
        tta_transforms: transforms.py의 get_tta_transforms() 반환 리스트
    """

    def __init__(self, model: nn.Module, tta_transforms: list):
        super().__init__()
        self.model = model
        self.tta_transforms = tta_transforms

    def forward(self, images_pil: list) -> torch.Tensor:
        """
        Args:
            images_pil: PIL Image 리스트 (배치)

        Returns:
            Tensor (B, 14) — TTA averaged probabilities
        """
        all_probs = []
        device = next(self.model.parameters()).device

        for transform in self.tta_transforms:
            batch = torch.stack([transform(img) for img in images_pil]).to(device)
            with torch.no_grad():
                probs = self.model(batch)
            all_probs.append(probs)

        return torch.stack(all_probs).mean(dim=0)


# ── 모델 팩토리 ───────────────────────────────────────────────────────────────

def build_model(
    model_name: str,
    pretrained: bool = True,
    num_classes: int = NUM_CLASSES,
) -> nn.Module:
    """
    모델 이름으로 CAD 모델 인스턴스 생성.

    Args:
        model_name: "densenet" | "efficientnet" | "vit"
        pretrained: ImageNet pretrained 가중치 사용 여부
        num_classes: 출력 클래스 수

    Returns:
        CAD 모델 인스턴스

    Raises:
        ValueError: 지원하지 않는 모델 이름
    """
    name = model_name.lower().strip()

    if name == "densenet":
        model = DenseNet121CAD(pretrained=pretrained, num_classes=num_classes)
    elif name in ("efficientnet", "efficientnet-b4"):
        model = EfficientNetCAD(pretrained=pretrained, num_classes=num_classes)
    elif name in ("vit", "vit-b/16", "vit_b_16"):
        model = ViTCAD(pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(
            f"지원하지 않는 모델: '{model_name}'. "
            f"지원 목록: {SUPPORTED_MODELS}"
        )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ 모델 생성: {model.model_name} ({n_params:.1f}M parameters)")
    return model


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> nn.Module:
    """
    저장된 체크포인트로부터 모델 가중치 로드.

    Args:
        model: 가중치를 로드할 모델 인스턴스
        checkpoint_path: .pt 또는 .pth 파일 경로
        device: 로드할 디바이스

    Returns:
        가중치가 로드된 모델
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # state_dict 키 유형에 따른 처리
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    print(f"✅ 체크포인트 로드 완료: {checkpoint_path}")
    return model


def get_model_info() -> Dict[str, Dict]:
    """
    UI 표시용 모델 정보 딕셔너리 반환.
    """
    return {
        "densenet": {
            "display_name": "DenseNet-121",
            "description": "Dense connectivity로 gradient vanishing 완화. 파라미터 효율적.",
            "params": "~8M",
            "input_size": "224×224",
            "strength": "안정적인 성능, 빠른 추론",
            "icon": "🔗",
        },
        "efficientnet": {
            "display_name": "EfficientNet-B4",
            "description": "Compound scaling으로 정확도/효율 균형 최적화.",
            "params": "~19M",
            "input_size": "224×224",
            "strength": "높은 정확도, 적절한 속도",
            "icon": "⚡",
        },
        "vit": {
            "display_name": "ViT-B/16",
            "description": "Self-attention으로 전역 문맥 포착. 대규모 데이터에서 강력.",
            "params": "~86M",
            "input_size": "224×224",
            "strength": "전역 패턴 학습, 최고 표현력",
            "icon": "🧠",
        },
    }


# ── 빠른 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fake = torch.randn(2, 3, 224, 224)

    for name in SUPPORTED_MODELS:
        m = build_model(name, pretrained=False)
        m.eval()
        with torch.no_grad():
            out = m(fake)
        print(f"  {m.model_name}: input {tuple(fake.shape)} → output {tuple(out.shape)}")
        assert out.shape == (2, NUM_CLASSES), f"출력 shape 불일치: {out.shape}"
        assert out.min() >= 0.0 and out.max() <= 1.0, "확률 범위 벗어남"

    print("\n✅ 모든 모델 sanity check 통과")
