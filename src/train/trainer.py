"""
CXR-CAD 학습 루프 뼈대.

실제 학습은 Google Colab 노트북 (notebooks/04_Training.ipynb)에서 수행합니다.
학습 완료 후 생성된 .pth 파일을 checkpoints/ 디렉토리에 배치하면
API 서버(api/main.py)가 자동으로 로드합니다.

구현 예정 기능:
  - Trainer      : 학습 루프, 검증 루프, 체크포인트 저장
  - EarlyStopping: validation AUROC 기준 조기 종료
"""

from __future__ import annotations

# TODO: Colab 노트북에서 구현 후 이 파일에 이식합니다.


class EarlyStopping:
    """
    Validation metric 기준 Early Stopping.

    Args:
        patience : 개선 없을 때 허용 에폭 수
        min_delta: 개선으로 인정할 최소 변화량
        mode     : 'max' (AUROC 등) | 'min' (loss 등)
    """

    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, metric: float) -> bool:
        """
        Returns:
            True: 학습 중단, False: 계속
        """
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "max":
            improved = metric > self.best + self.min_delta
        else:
            improved = metric < self.best - self.min_delta

        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                return True
        return False


class Trainer:
    """
    CXR-CAD 모델 학습 루프 (뼈대).

    TODO: Colab(04_Training.ipynb)에서 아래 기능을 구현합니다.
      - train_one_epoch(): 배치 학습, 손실 계산, gradient update
      - validate()       : 검증 AUROC 계산
      - save_checkpoint(): best model .pth 저장
      - fit()            : epoch 루프 + early stopping

    체크포인트 저장 포맷 (api/main.py 로드 호환):
        torch.save({
            "epoch"            : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auroc"        : best_auroc,
        }, "checkpoints/<model_key>_best.pth")
    """

    def __init__(self, model, optimizer, loss_fn, device, early_stopping=None):
        raise NotImplementedError(
            "Trainer는 Colab(04_Training.ipynb)에서 구현합니다."
        )
