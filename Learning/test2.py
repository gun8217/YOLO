# 1. 라이브러리 임포트
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# 2. 데이터 생성 (3차 함수로 변경)
np.random.seed(42)
x_np = np.linspace(-5, 5, 200)
y_np = 3 * x_np**3 + 3 * x_np**2 + x_np + 5 + np.random.normal(0, 20, size=x_np.shape)  # 노이즈 추가

# 3. 훈련/검증 데이터 분리
x_train_np, x_val_np, y_train_np, y_val_np = train_test_split(x_np, y_np, test_size=0.2, random_state=42)

x_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
x_val = torch.tensor(x_val_np, dtype=torch.float32).unsqueeze(1)
y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1)

# 4. 모델 정의
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 5. 손실 함수, 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 6. 학습
epochs = 2000
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_loss = criterion(val_pred, y_val)
        val_losses.append(val_loss.item())

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

# 7. 손실 시각화
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()

# 8. 모델 저장
torch.save(model.state_dict(), "cubic_model.pth")
print(" 모델이 'cubic_model.pth'로 저장되었습니다.")

# 9. 모델 로드
loaded_model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
loaded_model.load_state_dict(torch.load("cubic_model.pth"))
loaded_model.eval()
print(" 저장된 모델을 성공적으로 로드했습니다.")

# 10. 예측 시각화
x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
with torch.no_grad():
    y_test_pred = loaded_model(x_test).squeeze().numpy()

# 실제 3차 함수와 비교
x_true = x_test.squeeze().numpy()
y_true = 3 * x_true**3 + 3 * x_true**2 + x_true + 5

plt.plot(x_true, y_true, label='Train Loss', color='blue')
plt.plot(x_true, y_test_pred, label='Prediction Loss', color='red', linestyle='--')
plt.title("Training vs Prediction Loss")
plt.xlabel("Epoch")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

# 11. 모델 사용 예제
x_input = torch.tensor([[4.0]])
with torch.no_grad():
    y_output = loaded_model(x_input)
print(f" Predicted y for x=4.0: {y_output.item():.4f}")
