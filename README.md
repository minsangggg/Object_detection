# 객체인식 (Object Recognition)
YOLOv8 기반 **객체탐지/세그멘테이션/포즈추정** + Roboflow Inference API + Hugging Face Transformers **추론(pipeline) & IMDB 파인튜닝**까지, “모델 사용 → 커스텀 학습/튜닝 → 평가/시각화” 흐름을 실습한 저장소입니다.

## 진행과정
- **Roboflow Inference API 연동**: API Key 기반 추론 → JSON 결과 파싱 → OpenCV로 bbox/label 시각화
- **YOLOv8 실전 파이프라인**  
  - 사전학습 모델로 Detection/Segmentation/Pose 추론 및 결과 저장  
  - 커스텀 데이터셋 학습(train) → 검증(val) → **hyperparameter tuning(tune)** → 최적 파라미터 재학습
- **Transformers 활용**
  - `pipeline()`로 다양한 태스크 빠른 적용(감성분석/QA/NER/요약/이미지 캡셔닝 등)
  - `Trainer`로 **IMDB 이진분류 fine-tuning** → 모델 저장 → 직접 추론

## 성과 
- **YOLOv8 커스텀 탐지 모델**: 기본 대비 **Precision/Recall/mAP/Fitness 개선**(예: mAP@0.5 0.916 → 0.950, mAP@0.5:0.95 0.691 → 0.759)
- **숫자 인식(탐지) 모델**: 튜닝 적용 후 **큰 폭 개선**(예: mAP@50 0.7798 → 0.9558, mAP@50-95 0.4586 → 0.7748)

## 노트북
- `1.객체탐지_Robo1.ipynb` : Roboflow API 추론 + 시각화
- `2.객체탐지_Robo2.ipynb` : YOLOv8 Detection/Seg/Pose 추론 + 결과 저장
- `3.탐지모델_Yolo.ipynb` : 커스텀 학습/검증/튜닝 파이프라인
- `4.숫자인식_Yolo.ipynb` : 숫자 데이터 학습 + 튜닝 성능 비교
- `5.transformers (1).ipynb` : pipeline 감성분석
- `6.허깅페이스 (2).ipynb` : HF InferenceClient + pipeline 다태스크 + IMDB 파인튜닝

## 빠른 실행
### 사용한 라이브러리
```bash
pip install ultralytics opencv-python numpy pyyaml python-dotenv
pip install inference-sdk roboflow
pip install torch transformers datasets huggingface_hub
