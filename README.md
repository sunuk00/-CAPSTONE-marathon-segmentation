# Marathon Path Segmentation

마라톤 경로 분할을 위한 실험용 프로젝트입니다.
모델 교체, 손실 함수 조합, config 기반 실험 실행을 쉽게 하도록 구성되어 있습니다.

## 프로젝트 구조

```text
temp_project/
├── configs/
│   ├── unet.yaml
│   ├── unet_bce_iou.yaml
│   ├── unet_bce_dice.yaml
│   ├── deeplabv3.yaml
│   ├── predict_unet.yaml
│   └── predict_deeplabv3.yaml
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       └── images/
├── outputs/
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   ├── deeplabv3.py
│   │   ├── resunet.py
│   │   └── segformer.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── losses/
│   │   ├── bce.py
│   │   ├── iou.py
│   │   ├── dice.py
│   │   └── __init__.py
│   └── core/
│       ├── engine.py
│       ├── metrics.py
│       ├── utils.py
│       └── __init__.py
├── requirements.txt
└── README.md
```

## 현재 모델 지원 상태

- unet: 구현 완료
- deeplabv3: 구현 완료
- resunet: placeholder
- segformer: placeholder

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 형식

학습 데이터는 아래 구조를 권장합니다.

```text
data/train/
├── images/
└── masks/
```

- 이미지와 마스크는 파일명 stem이 같아야 짝이 맞춰집니다.
- 예: images/0001.png, masks/0001.png

## 실행 방법

### 1) 학습: config 파일 사용

```bash
python src/train.py --config configs/unet_bce_iou.yaml
```

또는 모듈 실행:

```bash
python -m src.train --config configs/unet_bce_iou.yaml
```

### 2) 학습: config + 일부 옵션만 CLI로 덮어쓰기

```bash
python src/train.py --config configs/unet_bce_iou.yaml --epochs 50 --lr 0.0005
```

우선순위는 다음과 같습니다.

- 코드 기본값
- config 값
- CLI에서 직접 전달한 값(최종 우선)

### 3) 추론: config 파일 사용

```bash
python src/predict.py --config configs/predict_unet.yaml
```

### 4) 추론: config + 일부 옵션 덮어쓰기

```bash
python src/predict.py --config configs/predict_deeplabv3.yaml --threshold 0.6
```

## 주요 옵션

### train.py

- config: YAML 경로
- model-name: unet, deeplabv3, resunet, segformer
- data-root: 학습 데이터 루트
- image-size: 입력 해상도
- loss-type: bce_iou 또는 bce_dice
- bce-weight, iou-weight, dice-weight, pos-weight: 손실 가중치
- output-dir: 체크포인트/로그 저장 위치

### predict.py

- config: YAML 경로
- model-path: 체크포인트 경로
- image-dir: 추론할 이미지 폴더
- output-dir: 마스크 저장 폴더
- image-size: 입력 해상도
- threshold: 이진 마스크 기준값

## 모델별 전처리

전처리 로직은 src/data/dataset.py에 모아두었습니다.

- 공통: RGB 변환, resize, [0, 1] 스케일
- deeplabv3: ImageNet mean/std 정규화 추가
- unet: [0, 1] 스케일 기본 전처리

학습과 추론 모두 동일한 모델별 전처리 함수를 사용합니다.

## 모델 출력 형식

모든 모델은 아래 출력 형식을 반드시 맞춰야 합니다.

- 출력 텐서 형태: (B, 1, H, W)
- 값 의미: sigmoid를 통과하기 전 raw logits
- H, W: 입력 이미지 해상도와 동일해야 함

현재 코드에서 이 형식을 기준으로 동작하는 이유는 다음과 같습니다.

- 학습: BCE/IoU/Dice 손실 계산이 logits 기반으로 구현되어 있음
- 추론: predict.py에서 logits에 sigmoid를 적용한 뒤 threshold로 이진 마스크 생성

모델별 동작 예시:

- U-Net: 기본적으로 입력과 동일 해상도 logits 출력
- DeepLabV3: 내부 stride로 출력 해상도가 달라질 수 있어, 모델 내부에서 입력 해상도로 interpolate 후 반환

최종 저장 마스크 형식:

- 파일 형식: png
- 픽셀 값: 0 또는 255 (uint8)
- 저장 해상도: 원본 테스트 이미지와 동일

## 로그와 산출물

학습 시:

- output-dir/model_epoch_XXX.pt
- output-dir/model_final.pt
- output-dir/training_log.json

추론 시:

- output-dir/*_mask.png
- output-dir/prediction_log.json

예측 마스크는 원본 테스트 이미지 해상도로 저장됩니다.

## config 파일 구성 예시

학습 config 예시:

```yaml
model_name: "unet"
data_root: "data/train"
image_size: 512
loss_type: "bce_iou"
bce_weight: 0.3
iou_weight: 0.7
pos_weight: 10.0
epochs: 30
output_dir: "outputs/unet_trained/bce_iou_512_30"
```

추론 config 예시:

```yaml
model_path: "outputs/unet_trained/model_final.pt"
image_dir: "data/test/images"
output_dir: "outputs/predictions/unet"
image_size: 512
threshold: 0.5
```

## 참고

- 한 모델에 대해 loss 조합별 preset을 분리해 두면 실험 재현과 비교가 쉬워집니다.
- 단일 config로 운영하고 싶다면 unet.yaml 하나만 유지해도 됩니다.
