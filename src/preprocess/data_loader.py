"""
NIH ChestX-ray14 Data Loader.

실제 데이터 로딩, Patient ID 기준 5-Fold GroupKFold Split,
데이터 누수 검증, pos_weight 계산을 제공합니다.

데이터 구조:
    data_root/
    ├── images/
    │   ├── 00000001_000.png
    │   └── ...
    └── Data_Entry_2017_v2020.csv
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset, DataLoader

# ── 상수 ─────────────────────────────────────────────────────────────────────

DISEASE_LABELS: List[str] = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia",
]

NUM_CLASSES = len(DISEASE_LABELS)

# NIH CSV 컬럼명
_COL_IMAGE    = "Image Index"
_COL_FINDING  = "Finding Labels"
_COL_PATIENT  = "Patient ID"
_COL_AGE      = "Patient Age"
_COL_SEX      = "Patient Gender"
_COL_VIEW     = "View Position"
_COL_FOLLOW   = "Follow-up #"


# ── CSV 파싱 ──────────────────────────────────────────────────────────────────

def load_nih_csv(csv_path: str) -> pd.DataFrame:
    """
    NIH Data_Entry_2017_v2020.csv를 읽어 멀티-핫 레이블 컬럼을 추가한 DataFrame 반환.

    Args:
        csv_path: CSV 파일 경로

    Returns:
        DataFrame with columns: Image Index, Patient ID, + 14 disease binary columns
    """
    df = pd.read_csv(csv_path)

    # 멀티-핫 인코딩: "Finding Labels" → 14개 binary 컬럼
    for disease in DISEASE_LABELS:
        df[disease] = df[_COL_FINDING].apply(
            lambda findings: 1.0 if disease in findings.split("|") else 0.0
        )

    return df


def compute_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    """
    불균형 데이터 보정을 위한 pos_weight 계산.
    pos_weight[i] = (음성 샘플 수) / (양성 샘플 수) per class.

    Args:
        df: load_nih_csv() 반환 DataFrame (학습 세트만 전달할 것)

    Returns:
        Tensor of shape (14,) for BCEWithLogitsLoss / FocalLoss
    """
    pos_counts = df[DISEASE_LABELS].sum(axis=0).values.astype(float)
    neg_counts = len(df) - pos_counts
    pos_weight = neg_counts / np.clip(pos_counts, 1, None)
    return torch.tensor(pos_weight, dtype=torch.float32)


# ── Train / Val / Test Split ──────────────────────────────────────────────────

def split_by_patient(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient ID 기준 Train/Val/Test Split (데이터 누수 방지).

    동일 환자의 이미지가 서로 다른 세트에 들어가지 않도록 보장합니다.

    Args:
        df: 전체 DataFrame
        test_ratio: 테스트 세트 비율 (default 0.15)
        val_ratio: 검증 세트 비율, 훈련 세트 대비 (default 0.15)
        random_state: Random seed

    Returns:
        (train_df, val_df, test_df)
    """
    rng = np.random.default_rng(random_state)

    # Patient 단위로 Shuffle 후 분할
    patients = df[_COL_PATIENT].unique()
    rng.shuffle(patients)

    n_test = max(1, int(len(patients) * test_ratio))
    n_val  = max(1, int(len(patients) * val_ratio))

    test_patients  = set(patients[:n_test])
    val_patients   = set(patients[n_test : n_test + n_val])
    train_patients = set(patients[n_test + n_val :])

    train_df = df[df[_COL_PATIENT].isin(train_patients)].reset_index(drop=True)
    val_df   = df[df[_COL_PATIENT].isin(val_patients)].reset_index(drop=True)
    test_df  = df[df[_COL_PATIENT].isin(test_patients)].reset_index(drop=True)

    return train_df, val_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> bool:
    """
    세 세트 간 Patient ID 중복이 없음을 검증.

    Returns:
        True if no leakage, raises AssertionError otherwise.
    """
    train_pts = set(train_df[_COL_PATIENT])
    val_pts   = set(val_df[_COL_PATIENT])
    test_pts  = set(test_df[_COL_PATIENT])

    assert train_pts.isdisjoint(val_pts),  "❌ 데이터 누수: Train ∩ Val 환자 존재"
    assert train_pts.isdisjoint(test_pts), "❌ 데이터 누수: Train ∩ Test 환자 존재"
    assert val_pts.isdisjoint(test_pts),   "❌ 데이터 누수: Val ∩ Test 환자 존재"

    print(
        f"✅ 데이터 누수 없음 확인\n"
        f"   Train: {len(train_df):,} images / {len(train_pts):,} patients\n"
        f"   Val  : {len(val_df):,} images / {len(val_pts):,} patients\n"
        f"   Test : {len(test_df):,} images / {len(test_pts):,} patients"
    )
    return True


def get_group_kfold_splits(
    train_df: pd.DataFrame,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Patient ID 기준 5-Fold GroupKFold Splits 반환.

    Args:
        train_df: 훈련 DataFrame (split_by_patient 결과)
        n_splits: Fold 수 (default 5)

    Returns:
        List of (train_idx, val_idx) tuples (array of row indices)
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df[_COL_PATIENT].values
    X = np.arange(len(train_df))

    splits = list(gkf.split(X, groups=groups))
    print(f"✅ {n_splits}-Fold GroupKFold Splits 준비 완료")
    return splits


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class NIHChestXrayDataset(Dataset):
    """
    NIH ChestX-ray14 PyTorch Dataset.

    Args:
        df: load_nih_csv() 반환 DataFrame (split 적용 후)
        images_dir: images/ 폴더 경로
        transform: torchvision / albumentations 변환 파이프라인
        return_meta: True이면 (image, label, meta_dict) 반환
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform=None,
        return_meta: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.return_meta = return_meta

        # 이미지 파일 존재 여부 경고 (처음 10개만)
        missing = [
            row[_COL_IMAGE]
            for _, row in self.df.head(10).iterrows()
            if not (self.images_dir / row[_COL_IMAGE]).exists()
        ]
        if missing:
            warnings.warn(
                f"⚠️  images_dir에 이미지 없음 (예시): {missing[:3]}\n"
                f"   경로 확인: {self.images_dir}",
                UserWarning,
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.images_dir / row[_COL_IMAGE]

        # 이미지 로드 (파일 없으면 빈 tensor 반환)
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.fromarray(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )

        if self.transform is not None:
            image = self.transform(image)
        else:
            import torchvision.transforms as T
            image = T.ToTensor()(image)

        # 멀티-핫 레이블
        label = torch.tensor(
            row[DISEASE_LABELS].values.astype(np.float32),
            dtype=torch.float32,
        )

        if self.return_meta:
            meta = {
                "image_index":    row[_COL_IMAGE],
                "patient_id":     row[_COL_PATIENT],
                "patient_age":    row.get(_COL_AGE, -1),
                "patient_sex":    row.get(_COL_SEX, "Unknown"),
                "view_position":  row.get(_COL_VIEW, "Unknown"),
            }
            return image, label, meta

        return image, label


# ── DataLoader Factory ────────────────────────────────────────────────────────

def create_dataloader(
    df: pd.DataFrame,
    images_dir: str,
    transform=None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = False,
    return_meta: bool = False,
) -> DataLoader:
    """
    NIHChestXrayDataset → DataLoader 팩토리.

    Args:
        df: 해당 split의 DataFrame
        images_dir: images/ 폴더 경로
        transform: 변환 파이프라인
        batch_size: 배치 크기
        num_workers: 병렬 로드 수
        shuffle: 학습 세트에 True
        return_meta: 메타데이터 함께 반환 여부

    Returns:
        DataLoader
    """
    dataset = NIHChestXrayDataset(
        df=df,
        images_dir=images_dir,
        transform=transform,
        return_meta=return_meta,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


# ── 고수준 팩토리 ─────────────────────────────────────────────────────────────

def build_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_transform=None,
    eval_transform=None,
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
) -> Dict[str, DataLoader]:
    """
    CSV 로드 → Split → 누수 검증 → DataLoader 생성 원스톱 함수.

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader,
         "pos_weight": Tensor, "train_df": DataFrame}
    """
    csv_path = os.path.join(data_root, "Data_Entry_2017_v2020.csv")
    images_dir = os.path.join(data_root, "images")

    df = load_nih_csv(csv_path)
    train_df, val_df, test_df = split_by_patient(df, test_ratio, val_ratio)
    verify_no_leakage(train_df, val_df, test_df)

    pos_weight = compute_pos_weight(train_df)

    return {
        "train":      create_dataloader(train_df, images_dir, train_transform, batch_size, num_workers, shuffle=True),
        "val":        create_dataloader(val_df,   images_dir, eval_transform,  batch_size, num_workers, shuffle=False),
        "test":       create_dataloader(test_df,  images_dir, eval_transform,  batch_size, num_workers, shuffle=False),
        "pos_weight": pos_weight,
        "train_df":   train_df,
        "val_df":     val_df,
        "test_df":    test_df,
    }
