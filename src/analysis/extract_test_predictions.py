import os
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.train.models import build_model, DISEASE_LABELS
from src.preprocess.nih_loader import build_data_loaders

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 모델 레이블 로드
    model_key = args.model
    model = build_model(model_key)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{model_key}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()

    # 2. 메타데이터 및 Test DataLoader 준비
    # 실제 환경의 batch size 조정
    _, _, test_loader = build_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 3. CSV 매핑을 위한 파일 오픈
    meta_path = os.path.join(args.data_dir, "Data_Entry_2017.csv")
    meta_df = pd.read_csv(meta_path)
    
    # Image Index 별 통계를 빠르게 가져오기 위한 딕셔너리화
    meta_dict = meta_df.set_index("Image Index").to_dict("index")

    results = []

    # 4. 추론 루프
    print(f"Evaluating {model_key} on the test set...")
    with torch.no_grad():
        for images, labels, image_names in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            # 예측 로직 (Logits -> Sigmoid)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = labels.cpu().numpy()
            
            # DataFrame Rows 구성
            for i in range(len(image_names)):
                img_name = image_names[i]
                prob = probs[i]
                true_label = labels[i]
                
                info = meta_dict.get(img_name, {})
                row = {
                    "Image Index": img_name,
                    "Patient Age": info.get("Patient Age", np.nan),
                    "Patient Gender": info.get("Patient Gender", "Unknown"),
                    "View Position": info.get("View Position", "Unknown")
                }
                
                # 정답(GT) 저장
                for j, cls in enumerate(DISEASE_LABELS):
                    row[f"{cls}_true"] = float(true_label[j])
                    
                # 예측 확률(Prob) 저장
                for j, cls in enumerate(DISEASE_LABELS):
                    row[f"{cls}_prob"] = float(prob[j])
                    
                results.append(row)

    # 5. CSV 저장
    out_df = pd.DataFrame(results)
    out_path = os.path.join(args.checkpoint_dir, "test_predictions.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n✅ Successfully saved predictions to {out_path}")
    print("이 파일을 로컬 컴퓨터의 checkpoints/ 폴더로 복사한 뒤, 로컬에서 Jupyter Notebook 들을 실행하세요!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="densenet", choices=["densenet", "efficientnet", "vit"])
    parser.add_argument("--data_dir", type=str, default="../data/nih", help="Path to Data_Entry_2017.csv and images")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints", help="Path to best.pth")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    main(args)
