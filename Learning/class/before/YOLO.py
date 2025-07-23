from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# YOLOv8 pose 모델 로드
model = YOLO('yolo11s-pose.pt')  # 자동으로 다운로드 및 실행
# model = YOLO('yolo11s.pt')

# 이미지 불러오기
image_path = 'C:/Users/602-17/YOLO/Learning/class/before/images/83185_28453_5355.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 추론 수행
results = model(image_rgb)

# 시각화
results[0].plot()  # 결과 이미지에 keypoints와 bbox 그리기

# 시각화된 결과 얻기 (바운딩박스 + 키포인트)
result_image = results[0].plot()

# 시각화 출력
plt.imshow(results[0].plot())
plt.axis('off')
plt.title('YOLO11 Detection')
plt.show()

# 결과 저장
cv2.imwrite('C:/Users/602-17/YOLO/Learning/class/before/images/YOLO11s_pose_result.jpg', result_image)
print("결과 이미지가 '_result.jpg'로 저장되었습니다.")