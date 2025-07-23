from custom_cnn import CustomCNN
import torch
import torchvision.transforms as transforms
import cv2

img = cv2.imread('C:/Users/602-17/YOLO/dental/data/cropped/test/caries/img102_0.jpg')

# # 모델 로드
# model = CustomCNN(num_classes=2)
# model.load_state_dict(torch.load('C:/Users/602-17/YOLO/dental/model/custom_cnn_weights.pth'))
# model.eval()

# # 이미지 전처리
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((160, 160)),
#     transforms.ToTensor()
# ])

# # 테스트용 crop 이미지
# img = cv2.imread('C:/Users/602-17/YOLO/dental/data/cropped/test/crop001.jpg')
# input_tensor = transform(img).unsqueeze(0)

# # 추론
# with torch.no_grad():
#     output = model(input_tensor)
#     predicted_class = torch.argmax(output, dim=1).item()
#     print("Predicted:", "normal" if predicted_class == 0 else "caries")
