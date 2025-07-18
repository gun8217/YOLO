import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 데이터 생성 (클래스 3개)
X, y = make_classification(n_samples=1000, n_features=4, n_classes=3,
                           n_informative=3, n_redundant=0, random_state=42)

# 2. 학습/검증 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 정규화
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 4. 텐서 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)   # CrossEntropyLoss는 long 타입 정수
y_val = torch.tensor(y_val, dtype=torch.long)

# 5. 모델 정의 (출력 노드 수 = 클래스 수)
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 3)  # 클래스 수 = 3
)

# 6. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()  # 내부에 softmax 포함
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 7. 학습
epochs = 100
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)            # shape: (batch_size, 3)
    loss = criterion(output, y_train)  # CrossEntropyLoss expects raw logits + long labels
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    # 검증
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        val_losses.append(val_loss.item())

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# 8. 시각화
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Multiclass Classification Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# 9. 정확도 평가
with torch.no_grad():
    logits = model(X_val)     # Softmax를 적용하지 않았으므로 확률이 아닌 값 리턴
    preds = torch.argmax(logits, dim=1)    # 가장 높은 확률의 클래스 인덱스 리턴
    acc = (preds == y_val).float().mean()  # 정답을 대상으로 평균 계산
    print(f" Validation Accuracy: {acc.item() * 100:.2f}%")




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
epochs = 300
loss_history = []
for epoch in range(epochs):
    model.train()
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