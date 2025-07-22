import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# from CNN_Dataset_test import model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE=", DEVICE)

# transform = transforms.Compose([
#     transforms.Resize((128, 128)),      # 이미지를 고정 크기로 설정
#     transforms.ToTensor(),              # 이미지를 PyTorch 텐서로 변환
#     transforms.Normalize([0.5], [0.5])  # 빠르고 안정적인 학습을 위한 정규화(0~1 -> -1~1), (x-0.5)/0.5
# ])

# label_map = {'BAD': 0, 'GOOD': 1}
# class_names = list(label_map.keys())

model.load_state_dict(torch.load("carrot_cnn.pth", map_location=DEVICE))
model.eval()


# 8. 실제 이미지 예측 함수
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    output = model(image_tensor)
    pred = output.argmax(1).item()
    plt.imshow(np.array(image))
    plt.title(f"Prediction: {class_names[pred]}")
    plt.axis('off')
    plt.show()

# 9. 예측 실행 예시
predict_image(data_path+'/val/cat/cat1.jpg')