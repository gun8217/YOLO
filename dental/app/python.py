from ultralytics import YOLO

model = YOLO('C:/Users/602-18/YOLO/dental/yolov8n.pt')
model.train(data='C:/Users/602-18/YOLO/dental/app/dental.yaml', epochs=100, imgsz=320)