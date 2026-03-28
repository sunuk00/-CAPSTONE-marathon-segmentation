## 📊 실험 결과 및 히스토리 (Experiment Logs)

이 프로젝트의 모델 성능 개선 과정과 실험 결과를 기록합니다. 
성능 평가는 **Validation Dice Score**를 최우선 지표로 사용합니다.

### 📈 요약 테이블 (Summary)

| Exp ID | 날짜 | 모델 구조 | Loss 함수 | Img Size | Batch/Epoch | Val Dice | IoU | 비고 / 상태 |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |:--- |
| `exp01_baseline` | 26-03-28 | 기본 U-Net | BCEWithLogits | 512x512 | 4 / 30 | **0.0000** | N/A | 🔴 실패: All-Black 예측 (클래스 불균형) |
| `exp01b_baseline_rerun` | 26-03-28 | 기본 U-Net | BCEWithLogits | 512x512 | 기본 / 20 | **0.0000** | **0.0000** | 🔴 실패: 추론 결과가 거의 전부 흰색 배경(과검출) |
| `exp02_diceloss` | 진행 예정 | 기본 U-Net | BCE + **Dice Loss** | 512x512 | 4 / 30 | - | - |🟡 예정: Loss 변경으로 불균형 해소 시도 |
| `exp03_dilation` | 기획 중 | 기본 U-Net | BCE + Dice Loss | 512x512 | 4 / 30 | - | - | ⚪️ 대기: 정답 마스크 두께(Dilation) 확대 적용 |

<br>

### 📝 상세 실험 리뷰 (Detailed Reviews)

#### 🧪 [exp01] Baseline U-Net 실험 (26-03-28)
* **목적:** 베이스라인 코드 정상 동작 확인 및 초기 성능 측정
* **현상:** Train/Val Loss는 0.03~0.04대까지 안정적으로 감소했으나, Val Dice Score가 `0.0000`으로 수렴하는 치명적인 문제 발생.
* **원인 분석 (Troubleshooting):** * 마라톤 경로(1) 픽셀이 전체 이미지(512x512) 대비 극도로 적은 **극심한 클래스 불균형(Class Imbalance)** 현상.
  * AI 모델이 단순히 모든 픽셀을 배경(0)으로 예측하는 꼼수(All-Black Mask)를 부려도 전체 정확도가 99% 이상 나오기 때문에 Loss가 낮게 계산됨.
* **Next Step:** * 단순 픽셀 정확도를 보는 BCE Loss 대신, 겹치는 영역 자체에 가중치를 두는 **Dice Loss (혹은 Focal Loss)를 결합하여 학습을 강제**할 예정 (`exp02`).

#### 🧪 [exp01b] Baseline U-Net 재실행 (26-03-28)
* **실행 명령:** `python .\src\train_unet_baseline.py --epochs 20 --image-size 512`
* **목적:** 동일 베이스라인에서 epoch 수를 조정해 재학습 후 예측 결과 재확인
* **관찰 결과:** 학습 후 추론 이미지가 거의 전부 흰색(전경 1)으로 출력되는 **과검출(Over-segmentation)** 현상 발생.
* **해석:** 기존의 All-Black 실패와 반대 방향으로 모델이 한쪽 클래스로 붕괴한 상태로 보이며, 현재 BCE 단독 학습에서는 불안정하게 한 클래스에 치우치는 문제가 지속됨.
* **후속 조치:**
  * `exp02`에서 BCE + Dice Loss 조합으로 양성/음성 불균형 완화
  * 추론 시 `--threshold` 스윕(예: 0.5/0.6/0.7)으로 민감도 점검
  * Validation Dice/IoU 기준으로 best checkpoint 재선정 후 재추론

---