# 딥러닝 기반 나사 결함 탐지 모델 성능 비교

나사(Screw) 이미지를 이용한 불량 검출 이진 분류 시스템. <br>졸업논문 연구 프로젝트.

## 개요

산업 현장에서 발생하는 나사 불량을 딥러닝 기반 이미지 분류로 자동 검출합니다.  
불량 미검출(False Negative)을 최소화하는 것이 핵심 목표이며, **F2-Score**를 주요 평가 지표로 사용합니다.

- **데이터셋**: Kaggle 'Screw Anomalies Detection' (480장)
- **입력**: 나사 이미지 (정상 / 불량 5개 유형)
- **출력**: Good(0) / Bad(1) 이진 분류
- **핵심 문제**: 클래스 불균형 (Good 325장 vs Bad 100장) + 불량 미검출 최소화

### 불량 유형

| 유형 | 설명 |
|------|------|
| Type 1 | 나사끝 휘어짐 / 끊어짐 |
| Type 2 | 나사머리 깨짐 |
| Type 3 | 나사목 깨짐 |
| Type 4 | 나사산 깨짐 (측면) |
| Type 5 | 나사산 깨짐 (상단) |

## 실험 결과

### 최종 테스트 성능 (Test set: 55장)

| 모델 | Precision | Recall | F2-Score | FPS | FASI |
|------|-----------|--------|----------|-----|------|
| **ResNet-18** | **0.9500** | **1.0000** | **0.9896** | **88.8** | **0.9927** |
| MobileNet-V2 | 0.8571 | 0.6316 | 0.6667 | 82.9 | 0.7468 |
| VGG-16 | 0.3333 | 0.5789 | 0.5046 | 24.2 | 0.4373 |

**ResNet-18**이 F2-Score, Recall, FASI 모든 지표에서 최고 성능을 기록하여 최종 모델로 선정되었습니다.

### Grid Search 챔피언 가중치

| 모델 | Best bad_weight |
|------|-------------------|
| ResNet-18 | 2.0 |
| MobileNet-V2 | 2.5 |
| VGG-16 | 5.0 |

## 모델

ImageNet pretrained 백본 3종을 비교 실험합니다. 헤드 레이어만 교체 후 전체 파라미터 파인튜닝.

| 모델 | 교체 레이어 | 파라미터 수 |
|------|------------|------------|
| ResNet-18 | `fc = Linear(512, 2)` | 11.2M |
| MobileNet-V2 | `classifier[1] = Linear(1280, 2)` | 3.4M |
| VGG-16 | `classifier[6] = Linear(4096, 2)` | 138M |

## 방법론

### 클래스 불균형 처리
- `CrossEntropyLoss(weight=[1.0, bad_weight])` — 불량 클래스 손실 가중치 부여
- `bad_weight`는 1.0~5.0 범위를 Grid Search로 탐색

### 학습 전략
- **Stratified 5-Fold Cross Validation** — 소규모 데이터셋 평가 신뢰도 확보
- **Best weight 선택**: `MAX mean F2` 기준
- **Early Stopping**: F2-Score 기준 (patience=10)
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-2)

### 평가 지표

```
F2-Score = 5 × Precision × Recall / (4 × Precision + Recall)
FASI = α·F2 + (1-α)·FPS_normalized   (α=0.7)
```

- **F2-Score**: Recall에 높은 가중치 — 불량 미검출 최소화
- **FASI**: 산업 현장 배포를 위한 성능·속도 복합 지표

## 실험 환경

- **CPU**: Intel Core Ultra 9 285K
- **GPU**: NVIDIA RTX 5080
- **Python**: 3.10.20 / **PyTorch**: 2.10.0 (cu128)

## 데이터셋 구조

```
data/new_k-fold_data/              (총 480장)
        ├── train/                 (총 425장)
        │   ├── good/    (325장)
        │   ├── type1/   (20장)
        │   ├── type2/   (20장)
        │   ├── type3/   (20장)
        │   ├── type4/   (20장)
        │   └── type5/   (20장)
        └── test/                  (총 55장)
            ├── good/    (36장)
            ├── type1/   (4장)
            ├── type2/   (4장)
            ├── type3/   (5장)
            ├── type4/   (3장)
            └── type5/   (3장)
```

type1~5는 학습 시 모두 `label=1 (bad)`로 병합됩니다.  
데이터 분할: Train 340 / Val 85 / Test 55

## 설치

```bash
pip install torch torchvision numpy scikit-learn pytorch-grad-cam matplotlib seaborn pandas openpyxl pillow
```

> Windows / Jupyter 환경에서는 `NUM_WORKERS = 0` 으로 설정하세요.

## 실행

`main.ipynb`를 Jupyter에서 순차 실행합니다.

```
main6.ipynb
  ├─ 데이터 로드 및 전처리
  ├─ Grid Search + 5-Fold CV (모델별)
  │    └─ bad_weight sweep → Champion weight 선정
  │    └─ 임계값 튜닝 (val softmax 확률 기반)
  ├─ 최적 weight·threshold로 테스트 평가
  └─ 시각화 (Confusion Matrix, Grad-CAM, Weight Sweep 비교)
```

결과물은 `outputs/run_YYYYMMDD_HHMMSS/` 에 자동 저장됩니다.

## 프로젝트 구조

```
screw_defect/
    ├── main.ipynb                 # 메인 실험 노트북
    ├── src/
    │   ├── dataset.py              # ScrewDataset, get_image_paths_and_labels(), transform
    │   ├── engine.py               # 추론 속도 측정, run_grid_search_with_kfold_cv()
    │   ├── model.py                # build_model_bin() (ResNet/MobileNet/VGG)
    │   ├── visualization.py        # Grad-CAM, Confusion Matrix, Weight Sweep 시각화
    │   └── utils.py                # set_seed(), make_run_dir()
    └── data/                       # 데이터셋
```
