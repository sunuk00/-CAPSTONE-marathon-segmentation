## Marathon Path Detection - Experiment Log

이 문서는 `outputs` 기준 실험 이력을 정리합니다.
현재 기준 핵심 평가지표는 Validation Dice이며, IoU를 함께 기록합니다.

### 1. Quick Summary

| Exp ID | Date | Train Script | Loss | Setting | Best Val Dice | Best Val IoU | Status |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: | :--- |
| exp01_baseline | 26-03-28 | `src/train_unet_baseline.py` | BCEWithLogits | 512, batch 4, 30ep | 0.0000 | - | 실패 (all-black 경향) |
| exp01b_baseline_rerun | 26-03-28 | `src/train_unet_baseline.py` | BCEWithLogits | 512, 기본 batch, 20ep | 0.0000 | 0.0000 | 실패 (all-white 경향) |
| exp02_bce_iou | 26-03-28 | `src/train_unet_bce_iou.py` | BCE + Soft IoU (0.5/0.5) | 512, batch 4, 20ep | 0.0000 | 0.0000 | 실패 지속 |
| exp02b_bce_iou_posw | 26-03-28 | `src/train_unet_bce_iou.py` | BCE(pos_weight=10) + Soft IoU (0.5/0.5) | 512, batch 4, 30ep | **0.1547** | **0.0916** | 개선 확인 |
| exp03_bce03_iou07_posw | 26-03-28 | `src/train_unet_bce_iou.py` | BCE(0.3) + Soft IoU(0.7) + pos_weight=10 | 512, batch 4, 30ep | **0.1759** | **0.1028** | 추가 개선 확인 |
| exp03b_bce05_iou05_posw20 | 26-03-28 | `src/train_unet_bce_iou.py` | BCE(0.5) + Soft IoU(0.5) + pos_weight=20 | 512, batch 4, 30ep | **0.1616** | **0.0945** | 개선 확인 (exp03 대비 낮음) |
| exp04_bce_dice_posw10 | 26-03-28 | `src/train_unet_bce_dice.py` | BCE(0.5) + Soft Dice(0.5) + pos_weight=10 | 512, batch 4, 30ep | **0.1529** | **0.0894** | 개선 확인 |
| exp04b_bce03_dice07_posw10 | 26-03-28 | `src/train_unet_bce_dice.py` | BCE(0.3) + Soft Dice(0.7) + pos_weight=10 | 512, batch 4, 30ep | **0.1470** | **0.0870** | 개선 확인 (exp04 대비 낮음) |

### 2. Key Findings

1. 클래스 불균형이 매우 심함
  - foreground ratio: `0.000708` (약 0.07%)
  - background ratio: 99.93%
2. BCE 단독 또는 BCE+IoU(무가중)에서는 쉽게 붕괴
  - all-black 또는 all-white 형태로 수렴
3. `pos_weight` 추가 후 지표가 0에서 유의미하게 상승
  - 0.5/0.5 기준 best checkpoint: epoch 29 (Val Dice 0.1547, Val IoU 0.0916)
4. BCE/IoU 가중치를 0.3/0.7로 조정 시 추가 개선
  - best checkpoint: epoch 28 (Val Dice 0.1759, Val IoU 0.1028)
5. `pos_weight`를 20으로 올린 0.5/0.5 실험도 개선은 있으나 최고점은 0.1616
  - exp03(0.1759)에는 미달, 다만 0.5/0.5 계열 대비 안정적 개선 확인
6. BCE+Dice(0.5/0.5, pos_weight=10)도 학습은 안정적으로 진행
  - best checkpoint: epoch 29 (Val Dice 0.1529, Val IoU 0.0894)
7. BCE+Dice(0.3/0.7, pos_weight=10)는 동작하나, 현재 로그 기준 0.5/0.5보다 낮음
  - 관측 최고 Val Dice 0.1470, Val IoU 0.0870

### 3. Latest Run (exp05_bce03_dice07_posw10)

실행 명령

```powershell
python .\src\train_unet_bce_dice.py --bce-weight 0.3 --dice-weight 0.7 --pos-weight 10 --epochs 30 --image-size 512
```

학습 후반 로그 (이번 실행)

```text
Epoch 024/30 | Train Loss: 0.6829, Train Dice: 0.2125, Train IoU: 0.1270 | Val Loss: 0.7018, Val Dice: 0.1373, Val IoU: 0.0809
Saved best checkpoint: outputs\unet_bce_dice\best_unet.pt (val_dice=0.1373)
Epoch 025/30 | Train Loss: 0.6755, Train Dice: 0.2163, Train IoU: 0.1304 | Val Loss: 0.6979, Val Dice: 0.1430, Val IoU: 0.0838
Saved best checkpoint: outputs\unet_bce_dice\best_unet.pt (val_dice=0.1430)
Epoch 026/30 | Train Loss: 0.6715, Train Dice: 0.2150, Train IoU: 0.1278 | Val Loss: 0.7004, Val Dice: 0.1149, Val IoU: 0.0676
Epoch 027/30 | Train Loss: 0.6636, Train Dice: 0.2384, Train IoU: 0.1442 | Val Loss: 0.6833, Val Dice: 0.1281, Val IoU: 0.0744
Epoch 028/30 | Train Loss: 0.6609, Train Dice: 0.2166, Train IoU: 0.1297 | Val Loss: 0.7019, Val Dice: 0.1046, Val IoU: 0.0598
Epoch 029/30 | Train Loss: 0.6494, Train Dice: 0.2448, Train IoU: 0.1474 | Val Loss: 0.6986, Val Dice: 0.1049, Val IoU: 0.0599
Epoch 030/30 | Train Loss: 0.6406, Train Dice: 0.2456, Train IoU: 0.1492 | Val Loss: 0.6796, Val Dice: 0.1470, Val IoU: 0.0870
```

해석

1. BCE+Dice(0.3/0.7, pos_weight=10) 설정에서 관측 최고 Val Dice 0.1470
2. 이번 실행에서는 후반 변동성이 커서 epoch별 성능 편차가 큼
3. BCE+Dice 계열 내에서는 이전 exp04(0.1529)가 더 높고, 전체 최고는 여전히 exp03(0.1759)

### 4. Current File Structure (Training/Inference)

1. Baseline
  - Train: `src/train_unet_baseline.py`
  - Predict: `src/predict_unet_baseline.py`
2. BCE+IoU 계열
  - Train: `src/train_unet_bce_iou.py`
  - Predict: `src/predict_unet_bce_iou.py`
  - Output: `outputs/unet_bce_iou`
3. BCE+Dice 계열
  - Train: `src/train_unet_bce_dice.py`
  - Output: `outputs/unet_bce_dice`

### 5. Next Minimal Experiments

1. BCE+Dice 가중치 sweep
  - 0.5/0.5, 0.4/0.6, 0.3/0.7 비교
2. BCE+Dice `pos_weight` sweep
  - 10, 15, 20 비교
3. inference threshold sweep
  - 0.35, 0.4, 0.45, 0.5 비교
4. BCE+Dice (0.3/0.7) 설정 2회 재학습
  - seed 고정 상태에서 변동성 확인