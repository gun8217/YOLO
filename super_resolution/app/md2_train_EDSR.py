if __name__ == "__main__":
    import yaml
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from dataset import SRDataset
    from md2_EDSR import EDSR
    from pathlib import Path
    from torchvision.transforms import ToTensor
    from tqdm import tqdm

    # ───────────────────────
    # 설정 파일 로드
    with open("C:/Users/602-18/YOLO/super_resolution/app/config.yml", "r") as f:
        config = yaml.safe_load(f)

    # ───────────────────────
    # 하이퍼파라미터
    scale = config["scale"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    lr = config["learning_rate"]
    num_workers = config["num_workers"]

    # 디렉토리 설정
    BASE_DIR = Path(__file__).parent.parent
    HR_DIR = BASE_DIR / 'data/train_HR_gen'
    LR_DIR = BASE_DIR / 'data/train_LR_gen'
    MODEL_DIR = BASE_DIR / 'model'
    save_path = MODEL_DIR / 'edsr.pth'

    # ───────────────────────
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ───────────────────────
    # 데이터 로더
    train_dataset = SRDataset(LR_DIR, HR_DIR, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # ───────────────────────
    # 모델, 손실 함수, 옵티마이저
    model = EDSR(scale_factor=scale).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ───────────────────────
    # 학습 루프
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            preds = model(lr_imgs)
            loss = criterion(preds, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch} completed. Avg Loss: {epoch_loss / len(train_loader):.6f}")

    # ───────────────────────
    # 모델 저장
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"모델 저장 완료: {save_path}")
