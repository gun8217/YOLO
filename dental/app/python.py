from ultralytics import YOLO

# train6
# model = YOLO('C:/Users/602-18/YOLO/dental/yolov8n.pt')
# model.train(data='C:/Users/602-18/YOLO/dental/app/dental.yaml', epochs=50, imgsz=640)

# train7
# model = YOLO('C:/Users/602-18/YOLO/dental/yolov8n.pt')
# model.train(data='C:/Users/602-18/YOLO/dental/app/dental.yaml', epochs=100, imgsz=320)

# # train8 - epochs 60~70 범위로 하향 조정해도 무방
# model = YOLO('C:/Users/602-18/YOLO/dental/yolov8n.pt')
# model.train(data='C:/Users/602-18/YOLO/dental/app/dental.yaml', epochs=100, imgsz=512)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = YOLO('C:/Users/602-18/YOLO/dental/yolov8s.pt')
model.train(data='C:/Users/602-18/YOLO/dental/app/dental.yaml', epochs=100, imgsz=512)
