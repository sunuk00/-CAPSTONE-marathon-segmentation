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

### 3. Latest Run (exp04_bce05_iou05_posw20)

실행 명령

```powershell
python .\src\train_unet_bce_iou.py --bce-weight 0.5 --iou-weight 0.5 --pos-weight 20 --epochs 30 --image-size 512
```

학습 후반 로그 (이번 실행)

```text
Epoch 020/30 | Train Loss: 0.5480, Train Dice: 0.1915, Train IoU: 0.1138 | Val Loss: 0.5492, Val Dice: 0.1431, Val IoU: 0.0815
Saved best checkpoint: outputs\unet_bce_iou\best_unet.pt (val_dice=0.1431)
Epoch 021/30 | Train Loss: 0.5456, Train Dice: 0.1803, Train IoU: 0.1052 | Val Loss: 0.5545, Val Dice: 0.0734, Val IoU: 0.0400
Epoch 022/30 | Train Loss: 0.5433, Train Dice: 0.1808, Train IoU: 0.1060 | Val Loss: 0.5437, Val Dice: 0.1276, Val IoU: 0.0722
Epoch 023/30 | Train Loss: 0.5403, Train Dice: 0.1976, Train IoU: 0.1166 | Val Loss: 0.5451, Val Dice: 0.1246, Val IoU: 0.0707
Epoch 024/30 | Train Loss: 0.5342, Train Dice: 0.2077, Train IoU: 0.1237 | Val Loss: 0.5402, Val Dice: 0.1359, Val IoU: 0.0791
Epoch 025/30 | Train Loss: 0.5310, Train Dice: 0.2134, Train IoU: 0.1270 | Val Loss: 0.5383, Val Dice: 0.1354, Val IoU: 0.0794
Epoch 026/30 | Train Loss: 0.5287, Train Dice: 0.2020, Train IoU: 0.1188 | Val Loss: 0.5376, Val Dice: 0.1230, Val IoU: 0.0726
Epoch 027/30 | Train Loss: 0.5268, Train Dice: 0.2201, Train IoU: 0.1312 | Val Loss: 0.5339, Val Dice: 0.1317, Val IoU: 0.0752
Epoch 028/30 | Train Loss: 0.5253, Train Dice: 0.2209, Train IoU: 0.1327 | Val Loss: 0.5322, Val Dice: 0.1430, Val IoU: 0.0821
Epoch 029/30 | Train Loss: 0.5222, Train Dice: 0.2146, Train IoU: 0.1271 | Val Loss: 0.5288, Val Dice: 0.1616, Val IoU: 0.0945
Saved best checkpoint: outputs\unet_bce_iou\best_unet.pt (val_dice=0.1616)
```

해석

1. pos_weight=20 적용으로 0.5/0.5 설정에서도 best Val Dice 0.1616까지 상승
2. 최고 성능은 epoch 29에서 관찰 (Val IoU 0.0945)
3. 전체 최고 기록은 여전히 exp03(0.3/0.7, Dice 0.1759)이므로, 현재 best checkpoint 기준은 exp03 유지가 합리적

### 4. Current File Structure (Training/Inference)

1. Baseline
  - Train: `src/train_unet_baseline.py`
  - Predict: `src/predict_unet_baseline.py`
2. BCE+IoU 계열
  - Train: `src/train_unet_bce_iou.py`
  - Predict: `src/predict_unet_bce_iou.py`
  - Output: `outputs/unet_bce_iou`

### 5. Next Minimal Experiments

1. `pos_weight` sweep
  - 15, 20, 25 비교
2. inference threshold sweep
  - 0.35, 0.4, 0.45, 0.5 비교
3. 0.5/0.5 + pos_weight=20 설정 2회 재학습
  - seed 고정 상태에서 변동성 확인