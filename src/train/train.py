"""
CXR-CAD 학습 파이프라인.

구현 내용:
- 5-Fold GroupKFold Cross Validation (Patient ID 기준)
- Early Stopping (patience 기반)
- Cosine Annealing LR Scheduler
- Fold별 AUROC 기록 및 Mean ± Std 계산
- 체크포인트 저장 (checkpoints/fold{k}_best.pt)
- Focal Loss gamma 파라미터 실험 지원 (gamma=0, 1, 2)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

from src.preprocess.data_loader import (
    build_dataloaders,
    get_group_kfold_splits,
    create_dataloader,
    DISEASE_LABELS,
)
from src.preprocess.transforms import get_train_transforms, get_inference_transforms
from src.train.models import build_model, NUM_CLASSES
from src.train.losses import build_loss


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    검증 손실이 patience 에포크 동안 개선되지 않으면 학습 중단.

    Args:
        patience: 개선 없이 기다릴 에포크 수
        min_delta: 개선으로 인정할 최소 변화량
        mode: 'min' (손실) 또는 'max' (AUROC 등 지표)
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        self.best_score = None
        self.counter = 0
        self.should_stop = False


# ── 단일 에포크 학습/검증 ─────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    scaler,  # torch.cuda.amp.GradScaler
) -> float:
    """학습 1 에포크. 평균 loss 반환."""
    model.train()
    total_loss = 0.0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            logits = model(images)
            # sigmoid는 모델 내부에서 처리하므로 logit-level로 loss 계산
            # → sigmoid 제거 후 raw logit 출력이 있는 경우를 위한 처리
            # 현재 모델은 sigmoid 포함이므로 역산
            logits_raw = torch.log(logits.clamp(1e-7, 1 - 1e-7) / (1 - logits.clamp(1e-7, 1 - 1e-7)))
            loss = loss_fn(logits_raw, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    검증/테스트 평가.

    Returns:
        {"loss": float, "auroc_mean": float, "auroc_per_class": List[float]}
    """
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        probs = model(images)
        logits_raw = torch.log(probs.clamp(1e-7, 1 - 1e-7) / (1 - probs.clamp(1e-7, 1 - 1e-7)))
        loss = loss_fn(logits_raw, labels)
        total_loss += loss.item()

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 클래스별 AUROC (양성 샘플 없으면 NaN → 제외)
    auroc_per_class = []
    for i in range(NUM_CLASSES):
        if all_labels[:, i].sum() > 0:
            try:
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                auroc_per_class.append(auc)
            except Exception:
                pass

    auroc_mean = float(np.mean(auroc_per_class)) if auroc_per_class else 0.0

    return {
        "loss":            total_loss / max(len(loader), 1),
        "auroc_mean":      auroc_mean,
        "auroc_per_class": auroc_per_class,
    }


# ── 5-Fold 학습 루프 ──────────────────────────────────────────────────────────

def run_cross_validation(
    data_root: str,
    model_name: str = "densenet",
    gamma: float = 2.0,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 5,
    num_workers: int = 4,
    checkpoint_dir: str = "checkpoints",
    device_str: str = "auto",
    image_size: int = 224,
) -> Dict:
    """
    5-Fold GroupKFold Cross Validation 실행.

    Args:
        data_root   : NIH 데이터셋 루트 (images/ + CSV)
        model_name  : "densenet" | "efficientnet" | "vit"
        gamma       : Focal Loss gamma (0, 1, 2)
        n_folds     : Fold 수
        epochs      : 최대 에포크
        batch_size  : 배치 크기
        lr          : 학습률
        weight_decay: AdamW weight decay
        patience    : EarlyStopping patience
        num_workers : DataLoader worker 수
        checkpoint_dir: 체크포인트 저장 폴더
        device_str  : "auto" | "cuda" | "cpu"
        image_size  : 입력 이미지 크기

    Returns:
        {
            "fold_aurocs": List[float],
            "mean_auroc": float,
            "std_auroc": float,
            "fold_results": List[Dict],
        }
    """
    # 디바이스 설정
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"\n🚀 학습 시작 — 모델: {model_name.upper()} | device: {device} | gamma: {gamma}")

    # 데이터 로드 및 Split
    dataloaders = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=get_train_transforms(image_size),
        eval_transform=get_inference_transforms(image_size),
    )
    train_df   = dataloaders["train_df"]
    val_df     = dataloaders["val_df"]
    pos_weight = dataloaders["pos_weight"].to(device)

    # GroupKFold Splits
    splits = get_group_kfold_splits(train_df, n_splits=n_folds)

    # 체크포인트 폴더 생성
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fold_aurocs = []
    fold_results = []

    # ── Fold 루프 ──────────────────────────────────────────────────────────────
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(splits):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        fold_train_df = train_df.iloc[fold_train_idx].reset_index(drop=True)
        fold_val_df   = train_df.iloc[fold_val_idx].reset_index(drop=True)

        images_dir = os.path.join(data_root, "images")
        fold_train_loader = create_dataloader(
            fold_train_df, images_dir,
            transform=get_train_transforms(image_size),
            batch_size=batch_size, num_workers=num_workers, shuffle=True,
        )
        fold_val_loader = create_dataloader(
            fold_val_df, images_dir,
            transform=get_inference_transforms(image_size),
            batch_size=batch_size, num_workers=num_workers, shuffle=False,
        )

        # 모델 및 최적화기
        model = build_model(model_name, pretrained=True).to(device)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
        loss_fn   = build_loss(gamma=gamma, pos_weight=pos_weight)
        early_stopping = EarlyStopping(patience=patience, mode="max")

        use_amp = device.type == "cuda"
        scaler  = torch.amp.GradScaler("cuda") if use_amp else None

        best_auroc = 0.0
        best_epoch = 0
        ckpt_path  = ckpt_dir / f"{model_name}_fold{fold_idx + 1}_gamma{int(gamma)}_best.pt"

        epoch_log = []

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_loss = train_one_epoch(model, fold_train_loader, optimizer, loss_fn, device, scaler)
            val_metrics = evaluate(model, fold_val_loader, loss_fn, device)
            scheduler.step()

            val_auroc = val_metrics["auroc_mean"]
            elapsed   = time.time() - t0

            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_AUROC={val_auroc:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.1f}s"
            )

            epoch_log.append({
                "epoch":      epoch,
                "train_loss": train_loss,
                "val_loss":   val_metrics["loss"],
                "val_auroc":  val_auroc,
            })

            # 최고 체크포인트 저장
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_epoch = epoch
                torch.save({
                    "fold":              fold_idx + 1,
                    "epoch":             epoch,
                    "model_name":        model_name,
                    "val_auroc":         val_auroc,
                    "model_state_dict":  model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, ckpt_path)

            # Early Stopping
            if early_stopping.step(val_auroc):
                print(f"  ⏹  Early Stopping @ epoch {epoch} (best={best_auroc:.4f} @ epoch {best_epoch})")
                break

        fold_aurocs.append(best_auroc)
        fold_results.append({
            "fold":       fold_idx + 1,
            "best_auroc": best_auroc,
            "best_epoch": best_epoch,
            "epoch_log":  epoch_log,
            "ckpt_path":  str(ckpt_path),
        })
        print(f"\n  ✅ Fold {fold_idx + 1} 완료 — Best AUROC: {best_auroc:.4f} @ epoch {best_epoch}")

    # ── 결과 요약 ─────────────────────────────────────────────────────────────
    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc  = float(np.std(fold_aurocs))

    print(f"\n{'='*60}")
    print(f"  {n_folds}-Fold CV 결과 요약")
    print(f"{'='*60}")
    for i, (fold_r, auc) in enumerate(zip(fold_results, fold_aurocs)):
        print(f"  Fold {i+1}: AUROC = {auc:.4f}")
    print(f"\n  Mean AUROC : {mean_auroc:.4f}")
    print(f"  Std  AUROC : {std_auroc:.4f}")
    print(f"  모델: {model_name.upper()} | gamma: {gamma}")

    return {
        "fold_aurocs":  fold_aurocs,
        "mean_auroc":   mean_auroc,
        "std_auroc":    std_auroc,
        "fold_results": fold_results,
    }


# ── CLI 인터페이스 ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CXR-CAD 5-Fold GroupKFold 학습")
    parser.add_argument("--data_root",      type=str, required=True,          help="NIH 데이터셋 루트 경로")
    parser.add_argument("--model",          type=str, default="densenet",      choices=["densenet", "efficientnet", "vit"])
    parser.add_argument("--gamma",          type=float, default=2.0,           help="Focal Loss gamma (0, 1, 2)")
    parser.add_argument("--n_folds",        type=int, default=5)
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--batch_size",     type=int, default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-5)
    parser.add_argument("--patience",       type=int, default=5)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device",         type=str, default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_cross_validation(
        data_root      = args.data_root,
        model_name     = args.model,
        gamma          = args.gamma,
        n_folds        = args.n_folds,
        epochs         = args.epochs,
        batch_size     = args.batch_size,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        patience       = args.patience,
        num_workers    = args.num_workers,
        checkpoint_dir = args.checkpoint_dir,
        device_str     = args.device,
    )
