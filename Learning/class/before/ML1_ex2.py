import matplotlib.pyplot as plt
import numpy as np

# 2D 함수: f(x, y) = x^2 + y^2
X, Y = np.meshgrid(np.linspace(-3, 3, 25), np.linspace(-3, 3, 25))
Z = X**2 + Y**2
grad_x = 2 * X
grad_y = 2 * Y

# 시각화
plt.figure(figsize=(7, 6))

# 등고선
contour = plt.contourf(X, Y, Z, levels=30, cmap='YlGnBu')
plt.colorbar(contour, label='f(x, y)')

# 그래디언트 벡터 표시
plt.quiver(X, Y, grad_x, grad_y, color='red', scale=30, label='Gradient (∇f)')

# 중심 표시
plt.scatter(0, 0, color='black', s=50, label='최솟값 위치 (0,0)')

# 타이틀 및 설정
plt.title("f(x, y) = x² + y²의 그래디언트 방향")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.show()
