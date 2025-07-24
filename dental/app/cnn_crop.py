import os
import cv2

# 경로 설정
image_dir = 'C:/Users/602-17/YOLO/dental/data/images/train'
label_dir = 'C:/Users/602-17/YOLO/dental/data/labels/train'
output_dir = 'C:/Users/602-17/YOLO/dental/data/cropped/train'

# 클래스 이름 매핑
class_map = {
    '0': 'normal',
    '1': 'cavity'
}

# 클래스별 폴더 생성
for class_name in class_map.values():
    os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

# 이미지 파일 순회
for filename in os.listdir(image_dir):
    if not filename.lower().endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

    if not os.path.exists(label_path):
        continue  # 라벨 없으면 스킵

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    with open(label_path, 'r') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # YOLO 형식이 아니면 스킵

            cls, x_center, y_center, box_w, box_h = map(float, parts)
            cls = str(int(cls))  # '0' or '1'

            # 바운딩 박스 좌표 계산
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            # 좌표 클리핑
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # crop
            roi = img[y1:y2, x1:x2]
            if roi.size == 0:
                continue  # 빈 이미지 스킵

            # 저장
            class_name = class_map.get(cls, 'unknown')
            save_name = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            save_path = os.path.join(output_dir, class_name, save_name)
            cv2.imwrite(save_path, roi)