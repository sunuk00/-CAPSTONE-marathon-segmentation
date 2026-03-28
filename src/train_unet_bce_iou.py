import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 마라톤 이미지와 마스킹 이미지 짝짓기 함수 
def collect_pairs(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    # 이미지와 마스크가 짝지어진 리스트를 저장할 빈 리스트 생성
    pairs: List[Tuple[Path, Path]] = [] 
    # 이미지 폴더에서 유효한 확장자를 가진 파일들만 리스트로 만듬
    image_paths = [p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]#

    # 이미지 파일들을 순서대로 처리하면서, 각 이미지에 대응하는 마스크 파일이 있는지 확인
    for image_path in sorted(image_paths):
        stem = image_path.stem
        candidates = [masks_dir / f"{stem}{ext}" for ext in VALID_EXTENSIONS]
        mask_path = next((p for p in candidates if p.exists()), None)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    return pairs


# Train과 Validation 데이터 분할 함수
def split_pairs(
    pairs: Sequence[Tuple[Path, Path]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]: # 
    pairs = list(pairs)
    rng = random.Random(seed) 
    rng.shuffle(pairs) # 랜덤 시드를 사용하여 데이터를 섞음 - 이렇게 하면 매번 같은 순서로 섞이게 되어, 실험의 재현성을 높여줌
    val_count = max(1, int(len(pairs) * val_ratio)) # 일정 비율로 분할, val 데이터는 최소 하나는 있도록 설정
    val_pairs = pairs[:val_count] # 섞인 리스트에서 앞부분을 validation 데이터로, 나머지를 training 데이터로 사용
    train_pairs = pairs[val_count:] 
    return train_pairs, val_pairs


class MarathonSegDataset(Dataset):

    # (이미지 경로, 마스크 경로) 리스트, image_size를 입력으로 받음 - image_size는 모델에 입력하기 전에 이미지를 조절할 크기
    def __init__(self, pairs: Sequence[Tuple[Path, Path]], image_size: int) -> None:
        self.pairs = list(pairs)
        self.image_size = image_size

    # 이미지-마스크 쌍의 개수 반환 - DataLoader가 이 함수를 사용하여 전체 데이터셋의 크기를 알 수 있도록 함
    def __len__(self) -> int:
        return len(self.pairs)
    
    ## Todo: 256 x 256으로 이미지를 모두 통일하여 축소하는지 확인, 만약 그렇다면 성능 향상을 위하여 512x512, 1024x1024를 고려해볼 수 있음 - 모델이 더 큰 이미지를 처리할 수 있도록 조정하고, 학습에 필요한 GPU 메모리가 충분한지 확인해야 함
    # 데이터 추출 및 전처리 함수 - DataLoader가 이 함수를 사용하여 특정 인덱스에 해당하는 이미지와 마스크를 불러오고, 모델이 학습할 수 있는 형태로 변환함
    def __getitem__(self, index: int):
        image_path, mask_path = self.pairs[index]

        # Pilow(PIL) 라이브러리로 Image를 열고, 이미지를 무조건 RGB 컬러 이미지로 통일 
        # 그리고 이미지를 지정된 크기로 조절, BILINEAR(쌍선형 보간법)을 사용하여 픽셀 사이를 부드럽게 채움 - 이렇게 하면 모델이 입력 이미지를 256x256 크기로 통일시켜 학습할 수 있도록 도와줌
        image = Image.open(image_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

        # 마스크 이미지를 흑백 이미지로 변환
        # NEAREST(최근점 이웃 보간법) 사용 - 마스크는 픽셀값이 0 또는 255인 이진 이미지이므로, 보간법을 사용하여 픽셀값이 중간값으로 변하는 것을 방지함
        mask = Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.Resampling.NEAREST
        )

        # PIL 이미지를 numpy 배열로 변환
        # 255.0으로 나눠줌으로써 0 ~ 255 사이 픽셀값을 0.0 ~ 1.0 사이로 정규화 -> 딥러닝 모델이 훨씬 빠르고 안정적으로 학습하게 해줌 - 모델이 작은 숫자 범위에서 더 잘 수렴하도록 도와줌
        image_arr = np.asarray(image, dtype=np.float32) / 255.0

        # 흑백 마스크 배열에서 픽셀값이 127보다 크면 True(경로), 작으면 False(배경)으로 이진화 -> 노이즈 제거 - 이렇게 하면 모델이 경로와 배경을 더 명확하게 구분할 수 있도록 도와줌
        mask_arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)

        # numpy 배열(img_arr)을 .from_numpy()를 통해 Tensor로 변환 = (256, 256, 3)
        # PyTorch 모델은 (C, H, W)를 요구하므로 permute()를 사용하여 (3, 256, 256) 으로 변경 / 0: 높이, 1: 너비, 2: 채널  
        image_tensor = torch.from_numpy(image_arr).permute(2, 0, 1)

        # 마스크 이미지는 흑백 이미지이기 때문에 (높이, 너비)만 있는 2차원 데이터
        # 따라서 unsqueeze(0)을 사용하여 크기가 1인 가짜 채널을 하나 끼워넣어줌 
        mask_tensor = torch.from_numpy(mask_arr).unsqueeze(0)
        return image_tensor, mask_tensor


# conv 블럭 class - U-Net 모델에서 반복적으로 사용되는 컨볼루션 레이어와 활성화 함수, 배치 정규화 등을 하나의 블럭으로 묶어서 정의함 - 이렇게 하면 모델 구조가 더 깔끔해지고, 코드의 재사용성이 높아짐
class ConvBlock(nn.Module):
    
    # 들어오는 채널 수와, 출력되는 채널 수를 미리 지정 - 예를 들어, U-Net의 첫 번째 ConvBlock에서는 in_channels=3(RGB 이미지), out_channels=32(모델이 학습할 특징의 수)로 설정됨
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        # nn.Sequential은 안에 있는 layer들을 순서대로 통과시키는 역할을 함
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # 실제 데이터 x(이미지 텐서)가 모델에 들어왔을 때, self.block에 x을 올려서 과정을 밟고 결과물을 반환 
    def forward(self, x: torch.Tensor) -> torch.Tensor: # 입력 텐서 x는 (batch size, in_channels, height, width) 형태의 4차원 텐서임 - 예를 들어, (4, 3, 256, 256) 크기의 배치가 들어올 수 있음
        return self.block(x)
    

# U-Net 모델 클래스 정의 - U-Net은 이미지 분할에 널리 사용되는 모델로, 인코더-디코더 구조를 가지고 있으며, 인코더에서 추출한 특징을 디코더에서 활용하는 Skip Connection이 특징임
class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()

        # U-Net Encoder: 아래로 내려갈 수록 채널 수 증가 - 이미지에서 점점 더 복잡한 특징을 추출하기 위해, 인코더의 각 단계마다 채널 수를 늘려감 - base_channels는 첫 번째 ConvBlock에서 사용할 채널 수를 지정하며, 이후 단계에서는 2배씩 증가함
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        # MaxPool2d는 이미지의 공간적 크기를 절반으로 줄이는 역할을 함 - 인코더에서 특징을 추출하면서 이미지의 크기를 줄여가며, 모델이 더 넓은 영역의 정보를 학습할 수 있도록 도와줌
        self.pool = nn.MaxPool2d(2)
        self.bridge = ConvBlock(base_channels * 8, base_channels * 16) # 가장 깊은 곳

        # U-Net Decoder: 압축 이미지 다시 확대 (Skip Connection 생각하기) - 디코더에서는 인코더에서 추출한 특징을 활용하여 이미지를 점점 더 원래 크기로 복원해 나감 - ConvTranspose2d는 업샘플링을 수행하여 이미지의 공간적 크기를 두 배로 늘리는 역할을 함
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Head: 최종 결과물 
        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)


    # 모델에 입력된 이미지 텐서 x가 U-Net을 통과하면서 어떻게 변하는지 정의하는 함수 - 모델이 실제로 데이터를 처리할 때 이 함수가 호출됨
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 인코더 단계: 이미지에서 특징을 추출하면서 공간적 크기를 줄여감 - e1, e2, e3, e4는 각 인코더 단계에서 추출된 특징 맵을 저장하는 변수임 - 이 특징 맵들은 디코더에서 Skip Connection으로 활용됨
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 가장 깊은 곳에서 특징을 더 추출하는 브릿지 단계 - e4에서 추출된 특징 맵을 MaxPool로 다운샘플링한 후, bridge ConvBlock을 통과시켜서 더 복잡한 특징을 학습함 - 이렇게 하면 모델이 이미지의 전반적인 구조와 패턴을 더 잘 이해할 수 있도록 도와줌
        b = self.bridge(self.pool(e4))

        # 디코더 단계: 업샘플링과 Skip Connection을 통해 이미지를 점점 더 원래 크기로 복원해 나감 - 각 단계에서 업샘플링된 특징 맵과 인코더에서 추출된 특징 맵을 채널 방향으로 연결(concatenate)하여, 모델이 인코더에서 추출한 정보를 활용할 수 있도록 함 - 이렇게 하면 모델이 경로와 배경을 더 정확하게 구분할 수 있도록 도와줌
        # Skip Connection 
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1) # 채널 방향으로 연결 - 예를 들어, d4가 (batch size, base_channels * 8, height, width) 크기이고, e4가 (batch size, base_channels * 8, height, width) 크기라면, torch.cat([d4, e4], dim=1)을 통해 (batch size, base_channels * 16, height, width) 크기의 텐서가 만들어짐
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
 
        # (batch size, 1, 256, 256) 크기의 마스킹 예측값 반환 - 각 픽셀에 대해 경로일 확률을 나타내는 값이 들어있음 - 모델의 출력은 raw logits 형태이므로, 나중에 BCEWithLogitsLoss와 함께 사용하여 손실을 계산할 때, 내부적으로 시그모이드 함수를 적용하여 확률로 변환한 후 Binary Cross Entropy Loss를 계산함
        return self.head(d1)


# Dice score
def dice_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    
    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


# IoU (Intersection over Union) score - Dice와 다르게 False Positive/Negative를 더 엄격하게 평가함
# IoU = 교집합 / 합집합 = (TP) / (TP + FP + FN) - Dice와 달리 교집합을 두 배하지 않음
def iou_score_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (preds * target).sum(dim=1)
    # Union = 두 집합의 원소 개수 합 - 교집합 (즉,합집합)
    union = preds.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


# Soft IoU loss for training (differentiable). Thresholding을 쓰지 않고 확률값 그대로 사용함.
def soft_iou_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (probs * target).sum(dim=1)
    union = probs.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


class BCEIoULoss(nn.Module):
    # 첫 실험을 단순화하기 위해 BCE와 IoU를 0.5:0.5로 고정
    def __init__(
        self,
        # Todo: bce랑 iou의 Loss 비율 조정하면서 성능 향상 시도해보기 - 예를 들어, bce_weight=0.5, iou_weight=0.5로 설정하여 두 손실이 동일한 비중으로 모델 학습에 기여하도록 할 수 있음
        bce_weight: float = 0.5, 
        iou_weight: float = 0.5,

        # Todo: pos_weight 조정하면서 성능 향상 시도해보기 - 예를 들어, pos_weight=10.0으로 설정하여 경로 픽셀의 오차가 배경 픽셀보다 10배 더 크게 반영되도록 할 수 있음 - 이렇게 하면 모델이 경로 픽셀을 더 잘 학습하도록 도와줄 수 있음
        pos_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        # pos_weight는 양성(경로) 픽셀을 더 크게 벌주기 위한 가중치
        # foreground가 매우 희소할 때(현재 데이터처럼) BCE가 배경 위주로 학습되는 것을 완화함
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits와 같은 device/dtype으로 맞춰서 CPU/GPU 혼용 에러를 방지
        pos_weight = self.pos_weight.to(device=logits.device, dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=pos_weight,
        )
        iou_loss = soft_iou_loss_from_logits(logits, target)
        return self.bce_weight * bce_loss + self.iou_weight * iou_loss

@dataclass
class EpochStats:
    loss: float
    dice: float
    iou: float

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> EpochStats:
    is_train = optimizer is not None # Optimizer가 들어오면 trainning, 아니면 validation 모드
    model.train(is_train)

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    count = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        # 이전 배치를 학습하면서 모델에 쌓인 Gradient를 지움
        if is_train:
            optimizer.zero_grad()

        # images를 보고 model이 예측 (Forward 실행)
        logits = model(images)

        # loss 계산
        loss = criterion(logits, masks)

        if is_train:
            loss.backward()  # gradient 계산
            optimizer.step() # optimizer를 사용하여 학습 (가중치 업데이트)

        with torch.no_grad():
            dice = dice_score_from_logits(logits, masks)
            iou = iou_score_from_logits(logits, masks)

        # 이번 batch에서 이미지가 몇 장 들어있는지 확인
        batch_size = images.size(0)
        # 가중 평균을 위해 이번 batch의 평균 loss에 이미지 장수를 곱함, dice 점수와 iou 점수에도 곱함
        running_loss += loss.item() * batch_size
        running_dice += dice.item() * batch_size
        running_iou += iou.item() * batch_size

        # 지금까지 몇장의 이미지 처리했는지 카운트
        count += batch_size

    return EpochStats(loss=running_loss / count, dice=running_dice / count, iou=running_iou / count)


# Argument 설정
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="U-Net training with BCE + Soft IoU loss for marathon path segmentation")
    parser.add_argument("--data-root", type=str, default="data/train")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--iou-weight", type=float, default=0.5)
    parser.add_argument("--pos-weight", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/unet_bce_iou")
    return parser


def main() -> None:
    # 터미널에서 입력받은 하이퍼파라미터(학습률, 배치사이즈 등)나 파일 경로 등의 설정값을 불러옴 
    args = build_arg_parser().parse_args()

    # 매번 같은 결과를 얻기 위해 시드 값을 고정함 - 랜덤 시드 고정은 모델의 초기 가중치, 데이터 섞는 순서 등에서 일관된 결과를 얻도록 도와줌
    set_seed(args.seed)

    # 학습에 사용할 images와 경로 이미지 masks, 그리고 학습된 모델이 저장될 출력 폴더의 경로를 설정
    data_root = Path(args.data_root)
    images_dir = data_root / "images"
    masks_dir = data_root / "masks"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 이미지나 마스크 폴더가 실제로 존재하지 않다면 에러를 발생시킴
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(f"Expected folders not found: {images_dir} and {masks_dir}")

    # 원본 이미지 <-> 정답 마스크 이미지를 짝지어서 리스트로 만듬
    pairs = collect_pairs(images_dir, masks_dir)
    if len(pairs) < 2:
        raise RuntimeError("Need at least 2 image-mask pairs to split train/val")

    # split_pairs를 통해 짝지어진 데이터를 Train과 Validation으로 나눔 - val_ratio에 따라 일정 비율로 나누고, seed를 사용하여 랜덤하게 섞어서 나눔
    train_pairs, val_pairs = split_pairs(pairs, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Total pairs: {len(pairs)} | Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    # 이미지를 AI가 이해할 수 있는 Tensor로 변환하고 사이즈를 맞추는 클래스 생성 - MarathonSegDataset 클래스는 이미지와 마스크를 불러와서 전처리한 후 Tensor로 반환하는 역할을 함
    train_ds = MarathonSegDataset(train_pairs, image_size=args.image_size)
    val_ds = MarathonSegDataset(val_pairs, image_size=args.image_size)

    # GPU를 사용할 때 데이터를 메모리에 더 빠르게 올리기 위한 설정, GPU가 없으면 False로 설정
    pin_memory = torch.cuda.is_available()

    # DataLoader: 전체 데이터를 batch_size만큼 쪼개서 모델에 공급해줌 - shuffle=True로 설정하여 매 epoch마다 데이터의 순서를 섞어서 모델이 특정 순서에 의존하지 않도록 함
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # cuda가 사용 가능한 경우 GPU를 사용하도록 설정하고, U-Net 모델을 생성하여 해당 장치로 이동시킴 - 이렇게 하면 모델이 GPU에서 학습할 수 있도록 함
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA (GPU: {gpu_name})")
    else:
        print("Using device: CPU")
    model = UNet().to(device)


    # 정답 마스크와 AI가 예측한 결과를 비교하여 Loss를 계산하는 함수
    # 첫 실험 변경: BCE 단독 대신 BCE + Soft IoU 조합으로 학습함
    # pos_weight를 사용해 경로(양성) 픽셀 오차를 더 크게 반영
    criterion = BCEIoULoss(
        bce_weight=args.bce_weight,
        iou_weight=args.iou_weight,
        pos_weight=args.pos_weight,
    )
    print(
        f"Loss config -> BCE:{args.bce_weight}, IoU:{args.iou_weight}, "
        f"pos_weight:{args.pos_weight}"
    )

    # 가중치 업데이트하는 optimizer 선택함 - Adam은 일반적으로 딥러닝 모델에서 좋은 성능을 보이는 최적화 알고리즘으로, 학습률을 자동으로 조정하여 빠르게 수렴하도록 도와줌
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    best_val_dice = -1.0
    # 지정한 epoch만큼 학습을 반복 - 매 epoch마다 train_loader와 val_loader를 사용하여 모델을 학습하고 검증함
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_stats.loss:.4f}, Train Dice: {train_stats.dice:.4f}, Train IoU: {train_stats.iou:.4f} | "
            f"Val Loss: {val_stats.loss:.4f}, Val Dice: {val_stats.dice:.4f}, Val IoU: {val_stats.iou:.4f}"
        )

        # Dice 점수 확인 후 성능이 더 좋아지면 모델의 상태를 최신화 
        if val_stats.dice > best_val_dice:
            best_val_dice = val_stats.dice
            checkpoint_path = output_dir / "best_unet.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_dice": val_stats.dice,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved best checkpoint: {checkpoint_path} (val_dice={best_val_dice:.4f})")


if __name__ == "__main__":
    main()
