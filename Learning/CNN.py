#### 합성곱 신경망(Convolutional Neural Network, CNN)
#    : 이미지, 영상, 음성 등 공간적 구조가 있는 데이터를 처리하기 위해 설계된 딥러닝 모델

### 구성 요소
# 합성곱 층 (Convolutional Layer)	이미지의 특징을 추출하는 필터(커널)를 적용
# 풀링 층 (Pooling Layer)	특징 맵을 축소해 계산량을 줄이고 중요한 정보만 유지
# ReLU 활성화 함수	비선형성을 추가하여 복잡한 패턴 학습 가능
# 완전 연결 층 (Fully Connected Layer)	추출된 특징을 기반으로 최종 분류 수행

### 대표 CNN 아키텍처
# LeNet-5 : 초기 손글씨 인식용 CNN
# AlexNet : 딥러닝 붐을 일으킨 이미지넷 챔피언
# VGGNet : 단순하고 깊은 구조로 유명
# ResNet : 잔차 연결로 매우 깊은 네트워크 가능
# Inception : 다양한 크기의 필터를 병렬로 사용