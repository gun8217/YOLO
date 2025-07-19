import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
iris = load_iris()
X = iris.data  # shape: [150, 4]
y = iris.target  # shape: [150,]

# 정규화
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 텐서 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 2. 모델 정의
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # 출력 노드: 3개 (클래스 수)
        )

    def forward(self, x):
        return self.net(x)

model = IrisNet()

# 3. 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 학습
epochs = 1000
loss_history = []
for epoch in range(epochs):
    # model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}")

# 5. 평가
model.eval()
with torch.no_grad():
    pred_test = model(X_test_tensor)
    predicted = torch.argmax(pred_test, dim=1)
    accuracy = accuracy_score(y_test, predicted.numpy())

print(f"\n 테스트 정확도: {accuracy * 100:.2f}%")

# 6. 손실 시각화
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()