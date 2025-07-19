import matplotlib.pyplot as plt
import numpy as np

# 1. 직선에서의 기울기 (slope)
x_line = np.linspace(-5, 5, 100)
y_line = 2 * x_line + 1  # 기울기 2, y절편 1

# 2. 2D 함수에서의 gradient 예시: f(x, y) = x^2 + y^2
X, Y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
Z = X**2 + Y**2
grad_x = 2 * X  # ∂f/∂x
grad_y = 2 * Y  # ∂f/∂y

# 시각화
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 직선의 기울기
axs[0].plot(x_line, y_line, label='y = 2x + 1', color='blue')
axs[0].set_title("직선에서의 기울기 (Slope)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].grid(True)
axs[0].legend()
axs[0].axhline(0, color='gray', lw=1)
axs[0].axvline(0, color='gray', lw=1)

# 오른쪽: 함수에서의 Gradient
axs[1].contour(X, Y, Z, levels=20, cmap='viridis')  # 등고선
axs[1].quiver(X, Y, grad_x, grad_y, color='red')     # 벡터
axs[1].set_title("곡면에서의 그래디언트 (Gradient)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].grid(True)

plt.tight_layout()
plt.show()
