이 폴더에 Colab에서 학습한 .pth 파일을 배치하세요.

파일명 규칙:
  densenet_best.pth
  efficientnet_best.pth
  vit_best.pth

Colab 저장 코드 예시:
  torch.save({
      "epoch"              : epoch,
      "model_state_dict"   : model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "val_auroc"          : best_auroc,
  }, "densenet_best.pth")

주의: .pth 파일은 .gitignore에 의해 Git 저장소에 포함되지 않습니다.
