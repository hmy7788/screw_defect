# Screw Defect Detection

나사(Screw) 이미지를 이용한 불량 검출 이진 분류 시스템. <br>
졸업논문 연구 프로젝트.

## 개요

산업 현장에서 발생하는 나사 불량을 딥러닝 기반 이미지 분류로 자동 검출합니다.  
불량 미검출(False Negative)을 최소화하는 것이 핵심 목표이며, 이를 위해 **F2-Score**를 주요 평가 지표로 사용합니다.

- **입력**: 나사 이미지 (정상 / 불량 5개 유형)
- **출력**: Good(0) / Bad(1) 이진 분류
- **핵심 문제**: 클래스 불균형 (Good 325장 vs Bad 100장) + 불량 미검출 최소화

## 모델

ImageNet pretrained 백본 3종을 비교 실험합니다.

| 모델 | 교체 레이어 | 파라미터 수 |
|------|------------|------------|
| ResNet-18 | `fc = Linear(512, 2)` | 11.2M |
| MobileNet-V2 | `classifier[1] = Linear(1280, 2)` | 3.4M |
| VGG-16 | `classifier[6] = Linear(4096, 2)` | 138M |

## 방법론

### 클래스 불균형 처리
`CrossEntropyLoss(weight=[1.0, bad_weight])`로 불량 클래스 손실을 가중하여 보정.  
`bad_weight`는 1.0~5.0 범위를 Grid Search로 탐색.

### 학습 전략
- **Stratified 5-Fold Cross Validation** — 소규모 데이터셋의 평가 신뢰도 확보
- **하이퍼파라미터(bad_weight) 선택**: `MAX mean F2`로 안정적인 weight 선정
- **조기 종료**: F2-Score 기준 (val loss 아닌 최적화 목표 기준)
- **Optimizer**: Adam (lr=1e-4)

### 평가 지표
```
F2-Score = 5 × Precision × Recall / (4 × Precision + Recall)
FASI = α × F2 + (1-α) × FPS_normalized   (α=0.7, 속도-성능 복합 지표)
```

## 데이터셋 구조

```
data/new_k-fold_data/
├── train/
│   ├── good/    (325장)
│   ├── type1/   (20장)
│   ├── type2/   (20장)
│   ├── type3/   (20장)
│   ├── type4/   (20장)
│   └── type5/   (20장)
└── test/
    ├── good/    (36장)
    ├── type1/   (4장)
    ├── type2/   (4장)
    ├── type3/   (5장)
    ├── type4/   (3장)
    └── type5/   (3장)
```

type1~5는 학습 시 모두 `label=1 (bad)`로 병합됩니다.

## 설치

```bash
pip install torch torchvision numpy scikit-learn pytorch-grad-cam matplotlib seaborn pandas openpyxl pillow
```

> Windows / Jupyter 환경에서는 `NUM_WORKERS = 0` 으로 설정하세요.

## 실행

`main.ipynb`를 Jupyter에서 순차 실행합니다.

```
main.ipynb
  ├─ 데이터 로드
  ├─ Grid Search + 5-Fold CV (모델별)
  │    └─ bad_weight sweep → Champion weight 선정
  ├─ 최적 weight로 최종 학습 및 테스트 평가
  └─ 시각화 (Confusion Matrix, Grad-CAM, Weight Sweep 비교)
```

결과물은 `outputs/run_YYYYMMDD_HHMMSS/` 에 자동 저장됩니다.

## 프로젝트 구조

```
screw_defect/
├── main.ipynb              # 메인 실험 노트북
├── src/
│   ├── dataset.py          # ScrewDataset, 전처리 transform
│   ├── engine.py           # 학습 루프, Grid Search, 추론 속도 측정
│   ├── model.py            # build_model_bin() (ResNet/MobileNet/VGG)
│   ├── visualization.py    # 시각화 함수 (Grad-CAM, Confusion Matrix 등)
│   └── utils.py            # set_seed(), make_run_dir()
└── data/                   # 데이터셋 (git 미포함)
```
