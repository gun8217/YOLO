import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

from custom_cnn import CustomCNN

# 🧪 하이퍼파라미터
num_classes = 2
batch_size = 32
epochs = 20
lr = 0.001

# 🖼️ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# 📁 데이터셋 로드
train_dataset = datasets.ImageFolder(
    root='C:/Users/602-17/YOLO/dental/data/cropped/train',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 🧠 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 🧠 모델 초기화
model = CustomCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 🚀 학습 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Time: {epoch_time:.2f} sec")

# 💾 모델 저장
torch.save(model.state_dict(), 'C:/Users/602-17/YOLO/dental/model/custom_cnn_weights.pth')
