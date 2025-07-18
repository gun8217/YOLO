from sklearn.datasets import fetch_california_housing
import pandas as pd

# 데이터셋 로드
data = fetch_california_housing(as_frame=True)
df = data.frame

# CSV 저장 (선택사항)
df.to_csv("california_housing.csv", index=False)
print(df.head())




import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# CSV 파일 로드
df = pd.read_csv("california_housing.csv")

# 특성과 타겟 선택 (다변수 가능)
X = df[['AveRooms', 'AveOccup', 'HouseAge']].values  # 3개의 특성
y = df['MedHouseVal'].values  # 타겟: 집값

# 정규화 (중요!), 평균=0, 표준편차=1, 빠른 수렴과 안정적인 학습
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))

# PyTorch tensor 변환
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 모델 정의
model = nn.Linear(X_tensor.shape[1], 1)  # 디폴트 모드 = train

# 손실 함수 및 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습
epochs = 2000
losses = []
for epoch in range(epochs):
    pred = model(X_tensor)
    loss = criterion(pred, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 학습된 파라미터 출력
print("\nWeights:", model.weight.data)
print("Bias:", model.bias.data)

# 손실 시각화
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 예측 수행
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()              # 정규화된 예측
    y_true = y_tensor.numpy()                     # 정규화된 실제값

# 역정규화 (원래 단위로)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_true_inv = scaler_y.inverse_transform(y_true)

# 평가 지표 계산
mse = mean_squared_error(y_true_inv, y_pred_inv)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
r2 = r2_score(y_true_inv, y_pred_inv)   # 결정계수(0~1)

print("\n 모델 평가 지표:")
print(f"MSE (평균 제곱 오차):      {mse:.4f}")
print(f"MAE (평균 절대 오차):      {mae:.4f}")
print(f"R² Score (설명력):         {r2:.4f}") # 모든 feature 사용 시 더 높아짐




# 예측
#새로운 입력값 (예: 방 5개, 평균 거주인원 3명, 집 나이 20년)
new_data = [[5.0, 3.0, 20.0]]  # shape: [1, 3]

# 1. 입력 정규화
new_data_scaled = scaler_X.transform(new_data)
new_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

# 2. 모델 예측
model.eval()  # 평가 모드로 전환 (필수는 아니지만 관례적으로)
with torch.no_grad():
    prediction = model(new_tensor)  # shape: [1, 1]

# 3. 예측값 역정규화
pred_value = scaler_y.inverse_transform(prediction.numpy())[0][0]

print(f"예측된 집값: {pred_value:.3f} (단위: 10만 달러)")



# 모델과 스케일러 저장 (필요 시 함께 저장하는 것이 중요)
torch.save(model.state_dict(), "linear_model.pth")

# Scaler도 함께 저장 (입출력 정규화 복원용)
import joblib  # joblib은 sklearn 모델/객체 저장에 최적화
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print(" 모델과 스케일러 저장 완료")




# 모델 로드
# 모델 구조 재정의 (학습 당시와 동일해야 함)
loaded_model = nn.Linear(3, 1)  # 입력 특성 3개
loaded_model.load_state_dict(torch.load("linear_model.pth"))
loaded_model.eval()  # 평가 모드 전환

# 스케일러 불러오기
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

print(" 모델과 스케일러 로드 완료")



# 로드된 모델을 사용하여 예측하기
# 새 데이터: 방 4개, 평균 거주 2.5명, 집 나이 15년
new_data = [[4.0, 2.5, 15.0]]

# 입력 정규화
new_scaled = scaler_X.transform(new_data)
new_tensor = torch.tensor(new_scaled, dtype=torch.float32)

# 예측
with torch.no_grad():
    pred = loaded_model(new_tensor)

# 결과 역정규화
pred_value = scaler_y.inverse_transform(pred.numpy())[0][0]

print(f"예측된 집값: {pred_value:.3f} (단위: 10만 달러)")