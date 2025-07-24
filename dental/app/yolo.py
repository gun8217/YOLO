if __name__ == "__main__":
    # YOLO 전이학습
    from ultralytics import YOLO

    # COCO로 사전학습된 모델 불러오기
    model = YOLO('yolo11s.pt').to('cuda')

    # 전이학습 수행
    model.train(  
        data='C:/Users/602-17/YOLO/dental/app/dental.yaml',
        epochs=20,
        imgsz=640,
        batch=16,
        project='C:/Users/602-17/YOLO/dental/',  # ←  학습결과 저장 루트경로
        name='runs/dental/transfer',  # 하위 디렉토리 명
        pretrained=True,  # default는 True
        patience = 5,
        # es_metiric='mAP50'
        verbose=True
    )