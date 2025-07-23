import os
from torch.utils.data import Dataset
from PIL import Image


def count_images_in_subfolders(root_dir):
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            image_count = len([
                file for file in os.listdir(class_path)
                if file.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            print(f"[📁] {class_name}: {image_count} images")
            

###### 1. 하이퍼파라미터 및 설정
import torch
from torchvision import transforms

BATCH_SIZE = 4
EPOCHS = 30
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE=", DEVICE)

# label_map 정의
label_map = {'BAD': 0, 'GOOD': 1}
class_names = list(label_map.keys())

# path 선언
data_path = "C:/Users/602-17/YOLO/Learning/class/before/dataset/carrot"
dest_root = "C:/Users/602-17/YOLO/Learning/class/before/dataset_split/carrot"
train_dir = os.path.join(dest_root, "train")
val_dir = os.path.join(dest_root, "val")


##### 2. 이미지 전처리
import cv2

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def extract_features(img_path):
    img = cv2.imread(img_path)
    avg_color = img.mean(axis=(0, 1))        # Green 값만 사용
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = np.var(gray)
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = sum([cv2.contourArea(c) for c in contours])
    h, w = img.shape[:2]
    aspect = w / h
    return np.array([avg_color[1], texture, area, aspect], dtype=np.float32)


###### 2-1. 이미지 분할
import shutil
import random

# 분할 저장 경로
# # 분할 비율
# split_ratio = [0.8, 0.19, 0.01]
# splits = ['train', 'val', 'test']

# # 대상 폴더 구조 생성: .../carrot/train/GOOD 등
# for split in splits:
#     for cls in classes:
#         split_cls_dir = os.path.join(dest_root, split, cls)
#         os.makedirs(split_cls_dir, exist_ok=True)

# # 이미지 분할 및 복사
# for cls in classes:
#     src_dir = os.path.join(data_path, cls)
#     if not os.path.isdir(src_dir):
#         print(f"[❌] 디렉토리 없음: {src_dir}")
#         continue

#     images = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
#     random.shuffle(images)

#     total = len(images)
#     train_end = int(split_ratio[0] * total)
#     val_end = train_end + int(split_ratio[1] * total)

#     split_files = {
#         'train': images[:train_end],
#         'val': images[train_end:val_end],
#         'test': images[val_end:]
#     }

#     for split, file_list in split_files.items():
#         for img in file_list:
#             src = os.path.join(src_dir, img)
#             dst = os.path.join(dest_root, split, cls, img)
#             os.makedirs(os.path.dirname(dst), exist_ok=True)
#             shutil.copy(src, dst)


###### 2-2. 이미지 증강
import numpy as np

# def add_noise(img, stddev=50):
#     arr = np.array(img).astype(np.float32)
#     noise = np.random.normal(0, stddev, arr.shape)
#     noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
#     return Image.fromarray(noisy_arr)

# def apply_affine(img):
#     width, height = img.size
#     coeffs = (1, 0.2, -10,   # a, b, c
#               0.1, 1, -5)    # d, e, f
#     return img.transform((width, height), Image.AFFINE, coeffs, resample=Image.BICUBIC)

# def apply_rotation(img, angle=4):
#     return img.rotate(angle, resample=Image.BICUBIC, expand=True).crop((0, 0, img.size[0], img.size[1]))

# def apply_random_crop_resize(img, crop_ratio=0.9):
#     w, h = img.size
#     crop_w, crop_h = int(w * crop_ratio), int(h * crop_ratio)
#     left = np.random.randint(0, w - crop_w + 1)
#     top = np.random.randint(0, h - crop_h + 1)
#     cropped = img.crop((left, top, left + crop_w, top + crop_h))
#     return cropped

# def augment_images_in_subfolders(root_dir):
#     for subfolder in os.listdir(root_dir):
#         subfolder_path = os.path.join(root_dir, subfolder)
#         if os.path.isdir(subfolder_path):
#             for filename in os.listdir(subfolder_path):
#                 if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     img_path = os.path.join(subfolder_path, filename)
#                     img = Image.open(img_path)

#                     # 증강 수행
#                     noisy = add_noise(img)
#                     affine = apply_affine(img)
#                     rotated = apply_rotation(img)
#                     cropped = apply_random_crop_resize(img)

#                     # 저장 (같은 폴더, 이름만 변경)
#                     base_name, ext = os.path.splitext(filename)
#                     noisy.save(os.path.join(subfolder_path, f"{base_name}_noisy{ext}"))
#                     affine.save(os.path.join(subfolder_path, f"{base_name}_affine{ext}"))
#                     rotated.save(os.path.join(subfolder_path, f"{base_name}_rotated{ext}"))
#                     cropped.save(os.path.join(subfolder_path, f"{base_name}_cropped{ext}"))

# augment_images_in_subfolders(train_dir)
# augment_images_in_subfolders(val_dir)


##### 3. 커스텀 데이터셋 정의
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, label_map, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = label_map

        for class_name, label in label_map.items():
            class_path = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    full_path = os.path.join(class_path, fname)
                    self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        features = extract_features(img_path)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return image, features_tensor, label


##### 4. 모델 정의(이미지 + 4가지 특징 통합)
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(32 * 32 * 32, 64),           # img resize : 128, 128
            nn.Linear((56 * 56 * 32) + 4, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, extra_feat):
        conv_out = self.conv(x)
        flat = torch.flatten(conv_out, 1)      # 배치 차원(0)을 유지하고 나머지를 모두 펼쳐 2차원 텐서로 변환
        combined = torch.cat((flat, extra_feat), dim=1)
        return self.fc(combined)


###### 5. 학습 준비
from torch.utils.data import DataLoader

train_dataset = CustomImageDataset(root_dir=train_dir, label_map=label_map, transform=transform)
valid_dataset = CustomImageDataset(root_dir=val_dir, label_map=label_map, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)       # 학습 데이터는 epoch마다 섞음
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)      # 검증 데이터는 섞을 필요 없음

model = SimpleCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


###### 6. 학습루프
import time

train_acc_list, val_acc_list = [], []

for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    correct, total, loss_total = 0, 0, 0

    for x, extra_feat, y in train_loader:
        x, extra_feat, y = x.to(DEVICE), extra_feat.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(x, extra_feat)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)
    train_acc = correct / total
    train_acc_list.append(train_acc)

    # 검증
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, extra_feat, y in valid_loader:
            x, extra_feat, y = x.to(DEVICE), extra_feat.to(DEVICE), y.to(DEVICE)
            outputs = model(x, extra_feat)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)
    val_acc = correct / total
    val_acc_list.append(val_acc)

    elapsed = time.time() - start_time
    print(f"Epoch {epoch+1} | Loss: {loss_total:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.2f}s")


###### 7. 시각화
import matplotlib.pyplot as plt

plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.show()

torch.save(model.state_dict(), "carrot_cnn.pth")