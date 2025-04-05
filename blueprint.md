# OCR 모델 Blueprint

## 1. 모델 구조

### 1.1 Detector (DBNet++)
- **Backbone**: ResNet-50
- **FPN**: Feature Pyramid Network
- **ASF**: Adaptive Scale Fusion
- **Detection Head**: Probability & Threshold Maps
- **입력 크기**: 1280x1280
- **출력**: 텍스트 영역의 확률 맵과 임계값 맵

### 1.2 Classifier (ConvNeXt + Transformer)
- **Backbone**: ConvNeXt-Base
- **Transformer**: 6 layers, 8 heads
- **입력 크기**: 256x256
- **출력**: 문자 클래스 예측

## 2. 디렉토리 구조

```
src/ocr_model/
├── detector/
│   ├── model.py        # DBNet++ 모델 정의
│   ├── train.py        # 검출기 학습 스크립트
│   └── utils/
│       ├── data_utils.py  # 데이터 처리 유틸리티
│       └── model_utils.py # 모델 관련 유틸리티
├── classifier/
│   ├── model.py        # ConvNeXt + Transformer 모델 정의
│   ├── train.py        # 분류기 학습 스크립트
│   └── utils/
│       ├── data_utils.py  # 데이터 처리 유틸리티
│       └── model_utils.py # 모델 관련 유틸리티
├── config/
│   └── config.py       # 설정 파일
└── utils/
    ├── data_utils.py   # 공통 데이터 처리 유틸리티
    └── model_utils.py  # 공통 모델 유틸리티
```

## 3. 데이터 처리

### 3.1 입력 데이터
- JSON 파일에서 이미지 경로와 텍스트 좌표 정보 로드
- 이미지 전처리:
  - Detector: 1280x1280 크기로 리사이징 (원본 비율 유지)
  - Classifier: 256x256 크기로 리사이징 (정사각형)

### 3.2 출력 데이터
- Detector:
  - Probability Map: 텍스트 영역 확률
  - Threshold Map: 이진화 임계값
- Classifier:
  - 문자 클래스 예측
  - 시퀀스 길이: 최대 25자

## 4. 학습 설정

### 4.1 Detector
- 배치 크기: 4
- 학습률: 1e-4
- 에포크: 100
- 웜업 에포크: 1
- 가중치 감소: 1e-4

### 4.2 Classifier
- 배치 크기: 32
- 학습률: 1e-4
- 에포크: 100
- 웜업 에포크: 1
- 가중치 감소: 1e-4

## 5. 평가 메트릭

### 5.1 Detector
- Precision
- Recall
- F1-score
- IoU (Intersection over Union)

### 5.2 Classifier
- Accuracy
- Character Error Rate (CER)
- Word Error Rate (WER)

## 6. 최적화

### 6.1 메모리 최적화
- 이미지 크기 조정
- 배치 크기 조정
- 메모리 효율적인 데이터 로딩

### 6.2 속도 최적화
- 병렬 처리
- GPU 가속
- 효율적인 데이터 파이프라인

## 7. 배포

### 7.1 모델 저장
- 체크포인트 저장
- 최적 모델 저장
- 모델 메타데이터 저장

### 7.2 추론
- 배치 처리
- 실시간 처리
- 에러 처리