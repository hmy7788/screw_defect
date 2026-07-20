# 신규 실험 핸드오버 문서 (새 세션/새 컴퓨터용)

> 이 문서 하나만 읽으면 어디까지 왔고 뭘 해야 하는지 전부 파악할 수 있도록 작성함.
> Claude에게 이 파일을 읽혀서 세션을 이어가면 됨.

---

## 1. 프로젝트 배경

### 기존 졸업논문 실험 (`main.ipynb`)
- 데이터: `data/new_k-fold_data/` (Kaggle screw 480장, good 361 / bad 119)
- 방법: ResNet-18 / MobileNet-V2 / VGG-16 이진분류(supervised), weighted CrossEntropy
- 평가: Precision / Recall / **F2-Score** (주지표) / FPS / FASI
- 결과: **ResNet-18 (weight=2.0) 최고** — Recall 1.0, F2 0.9896, FASI 0.9927
- 결과 파일: `results.csv`, `PORTFOLIO.md`, `README.md`

### 이번 신규 실험 목적 (`main2.py`)
기존 baseline(ResNet-18, weight=2.0)을 **그대로 계승**하되:
1. 데이터셋을 **MVTec AD `data/screw`** 로 교체 (공식 벤치마크)
2. 평가 지표를 **6개로 확장** → 픽셀 단위 위치추정 정량평가 추가
3. "분류기는 결함을 탐지하지만 위치는 잘 못 짚는다"를 수치로 증명
   → 이후 비지도 이상탐지(PatchCore) 고도화의 동기 데이터로 활용

**포트폴리오 서사**: baseline → 한계 발견 → 고도화 로 이어지는 성장 스토리의 **중간 연결고리**

---

## 2. 6개 평가지표

| # | 지표 | 계산 방법 | 단위 |
|---|------|-----------|------|
| 1 | Precision | `precision_score(pos_label=1)` | 이미지 |
| 2 | Recall | `recall_score(pos_label=1)` | 이미지 |
| 3 | F2-Score | `fbeta_score(beta=2, pos_label=1)` | 이미지 |
| 4 | Image AUROC | `roc_auc_score(y_true, softmax_prob_bad)` | 이미지 |
| 5 | Pixel AUROC | `roc_auc_score(mask_pixels, cam_pixels)` | 픽셀 |
| 6 | AUPRO | `anomalib.metrics.aupro._AUPRO` 사용 | 픽셀(영역) |

> **Pixel AUROC vs AUPRO**: Pixel AUROC는 픽셀 수가 많은 결함 유형에 편향. AUPRO는 결함 영역(connected component)별로 정규화해서 더 공정한 위치평가 지표.

---

## 3. 데이터셋 구조 (`data/screw` = MVTec AD screw)

```
data/screw/
├── train/
│   └── good/          ← 320장 (정상만, 결함 없음!)
├── test/
│   ├── good/          ← 41장
│   ├── manipulated_front/  ← 24장
│   ├── scratch_head/       ← 24장
│   ├── scratch_neck/       ← 25장
│   ├── thread_side/        ← 23장
│   └── thread_top/         ← 23장
└── ground_truth/      ← 결함 이미지별 마스크 (good 폴더 없음!)
    ├── manipulated_front/  ← 000_mask.png ~ 023_mask.png
    ├── scratch_head/
    ├── scratch_neck/
    ├── thread_side/
    └── thread_top/
```

**중요**:
- 이미지·마스크 크기: **1024×1024**, 마스크값: {0, 255}
- train에는 **결함이 없다** → 지도학습을 위해 데이터를 직접 재분할해야 함
- good 이미지는 ground_truth 폴더 자체가 없음 → 마스크 = 전부 0 (빈 마스크)

### 이미지↔마스크 매핑 규칙
```
이미지:  data/screw/test/scratch_head/000.png
마스크:  data/screw/ground_truth/scratch_head/000_mask.png
                ↑ test → ground_truth          ↑ 000.png → 000_mask.png
```

---

## 4. 설계 결정

### 4-1. 데이터 재분할 (지도학습용)
MVTec train에 결함이 없으므로 good+결함 전체를 새로 분할:

| 풀 | 구성 | 수 |
|----|------|----|
| good 전체 | train/good(320) + test/good(41) | 361장 |
| 결함 전체 | test의 5유형 전부 | 119장 |

→ **세부유형(6종) 기준 stratified split**, seed=42, `test_size=0.4`

기대 결과:
- train: good ≈ 216, 결함 ≈ 71 → 불균형 3:1 → **weight=2.0 타당**
- test: good ≈ 145, 결함 ≈ 48 → **결함 전부 마스크 보유 → 픽셀평가 가능**

> `test_size=0.4`는 결함 픽셀 평가 표본을 충분히 확보하기 위한 선택. 상단 상수로 노출해 조정 가능.

### 4-2. 학습 설정
- 모델: `build_model_bin('resnet18')` — ImageNet pretrained, fc→Linear(512,2)
- 손실: `CrossEntropyLoss(weight=[1.0, 2.0])`
- 옵티마이저: Adam lr=1e-4
- 에포크: 40
- Val 분할: train의 20%를 val로 → **val F2 기준 best checkpoint 복원** (졸업논문 방식 계승)
- Early stop: patience=10

### 4-3. Grad-CAM 위치추정 방식
분류기는 픽셀별 이상맵이 없다 → **Grad-CAM 히트맵을 이상맵 대신 사용**

```
ResNet-18 추론 → Grad-CAM(target=layer4[-1], class=1(bad))
→ grayscale CAM (224×224, float [0,1])
→ GT 마스크(1024×1024)를 val_transform과 동일하게 축소 → 224×224 정렬
→ pixel AUROC / AUPRO 계산
```

**핵심**: CAM과 마스크를 **동일한 기하변환(Resize256+CenterCrop224)** 으로 정렬해야 픽셀이 대응됨.

### 4-4. AUPRO 라이브러리
```python
# 공개 AUPRO는 안 됨 (Batch 필드 요구)
from anomalib.metrics.aupro import _AUPRO   # ← raw 텐서용 베이스 클래스 직접 사용
```
anomalib 2.5.0 기준 이 경로로 import 가능하고 작동 확인됨.

---

## 5. 디렉토리 구조 (목표)

```
screw_defect/
├── main.ipynb                # (기존) 졸업논문 — 건드리지 않음
├── main2.py                  # (신규) 이번 실험 드라이버 ← 아직 미작성
├── PLAN.md                   # 설계 요약
├── HANDOVER.md               # 이 파일
├── PORTFOLIO.md / README.md / results.csv   # 기존, 변경 없음
│
├── src/
│   ├── dataset.py            # 기존 (ScrewDataset, train_transform, val_transform)
│   ├── model.py              # 기존 (build_model_bin)
│   ├── engine.py             # 기존 (학습 루프 패턴 참고용)
│   ├── visualization.py      # 기존 (Grad-CAM 타깃레이어 참고용)
│   ├── utils.py              # 기존 (set_seed, make_run_dir)
│   │
│   ├── mvtec_data.py         # (신규) ← 지금 여기 작업 중
│   └── loc_eval.py           # (신규) ← 아직 미작성
│
├── data/
│   ├── new_k-fold_data/      # 기존 실험 데이터 — 이번엔 사용 안 함
│   └── screw/                # MVTec AD — 이번 실험 데이터
│
└── outputs/
    └── run_YYYYMMDD_HHMMSS/
        ├── figures/          # Grad-CAM 오버레이 저장
        ├── weights/          # best checkpoint (.pth)
        └── results_screw_mvtec.csv
```

---

## 6. 구현 순서 및 현재 진행 상태

### STEP 1: `src/mvtec_data.py` ← **여기까지 왔음, 미완성**

**완료된 것**:
```python
# good 이미지 경로 수집 (361장) — 완료
all_good_paths = []
# train/good 320장 + test/good 41장 전체 경로로 담김
# os.path.exists 확인까지 완료
```

**아직 안 한 것 (여기서부터 시작)**:

#### (1) 결함 경로 + 마스크 경로 수집
5유형 각각을 순회하면서 `(이미지경로, 유형이름, 마스크경로)`를 묶어야 함.

```
DEFECT_TYPES = ['manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top']
GT_DIR = 'data/screw/ground_truth'

마스크 경로 변환: '000.png' → '000_mask.png'
    힌트: os.path.splitext(file_name)[0] + '_mask.png'

반환 형태 추천: defect_records = [{'path': ..., 'type': ..., 'mask': ...}, ...]
```

검증: `len(defect_records) == 119` + `os.path.exists(mask_path)` 몇 개 찍어보기.

#### (2) stratified split
```python
from sklearn.model_selection import train_test_split

# good에는 label=0, defect에는 label=1 부여
# stratify 기준: 이진 label이 아니라 세부유형(6종) — good / 5유형
# 이유: test에 각 유형이 골고루 들어가야 유형별 픽셀 평가 가능

# seed=42, test_size=0.4
```

검증: train/test 개수, test 결함의 유형별 개수 출력.

#### (3) 함수화
지금은 스크립트 형태인데, `main2.py`에서 import해서 쓸 수 있게 함수로 감싸야 함:
```python
def load_mvtec_screw(data_root='./data/screw', test_size=0.4, seed=42):
    ...
    return train_records, test_records
    # record = {'path': str, 'label': int, 'type': str, 'mask': str or None}
```

---

### STEP 2: `main2.py` 학습부 (STEP 1 완료 후)

`src/engine.py`의 패턴을 참고해 직접 짜기.

흐름:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Windows OMP 에러 방지 — 반드시 맨 위

import sys; sys.path.append('.')
from src.mvtec_data import load_mvtec_screw
from src.model import build_model_bin
from src.dataset import ScrewDataset, train_transform, val_transform
from src.utils import set_seed, make_run_dir

# 1. 데이터 로드 + 분할
train_records, test_records = load_mvtec_screw()

# 2. ScrewDataset 생성
#    train_records에서 path, label만 뽑아서 ScrewDataset 사용 (기존 클래스 재사용)
#    train 80% / val 20% 추가 분할 → val F2로 best checkpoint 선정

# 3. 학습 루프
#    CrossEntropyLoss(weight=[1.0, 2.0])
#    Adam lr=1e-4, 40 epoch, early stop patience=10
#    val F2 기준 best state_dict 저장 → 학습 끝나면 복원

# 4. test 이미지 단위 평가
#    model.eval() + no_grad → softmax prob 수집
#    Precision / Recall / F2 / Image AUROC 계산·출력
```

---

### STEP 3: `src/loc_eval.py` Grad-CAM 픽셀평가 (STEP 2 완료 후)

**함정 포인트 목록** (직접 부딪혀볼 것):

| # | 함정 | 증상 | 힌트 |
|---|------|------|------|
| 1 | Grad-CAM이 동작 안 함 | RuntimeError 또는 CAM이 전부 0 | `torch.no_grad()` 블록 **안에서** 호출하면 안 됨 — grad가 필요 |
| 2 | CAM↔마스크 크기 불일치 | pixel AUROC가 이상한 값 | 마스크를 val_transform과 **동일하게** Resize(256)+CenterCrop(224), NEAREST 보간 |
| 3 | 공개 AUPRO 에러 | TypeError / shape 에러 | `from anomalib.metrics.aupro import _AUPRO` 로 우회 |
| 4 | good 이미지 AUPRO 포함 문제 | AUPRO 왜곡 | pixel AUROC/AUPRO는 **결함 이미지만** 대상으로 계산 |

Grad-CAM 핵심 코드 패턴:
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
targets = [ClassifierOutputTarget(1)]   # class 1 = bad

# no_grad 밖에서!
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]   # (224, 224)
```

마스크 전처리 (val_transform과 동일한 기하변환):
```python
from torchvision import transforms
from PIL import Image

mask_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(224),
])
# Image.open(mask_path).convert('L') → mask_transform → numpy → /255 → {0,1}
```

AUPRO 사용 패턴:
```python
from anomalib.metrics.aupro import _AUPRO
import torch

aupro_metric = _AUPRO()
# cam_tensor: (N, 1, 224, 224) float
# mask_tensor: (N, 1, 224, 224) int (0 or 1)
aupro_metric.update(cam_tensor, mask_tensor)
aupro_score = aupro_metric.compute()
```

---

### STEP 4: 결과 저장 (모든 지표 계산 후)

`results_screw_mvtec.csv` 형식:
```
experiment, model, weight, image_precision, image_recall, image_f2, image_auroc, pixel_auroc, aupro
mvtec_supervised, ResNet-18, 2.0, X.XXXX, X.XXXX, X.XXXX, X.XXXX, X.XXXX, X.XXXX
```

---

## 7. 재사용할 기존 src 모듈 요약

| 파일 | 재사용할 것 | 비고 |
|------|------------|------|
| `src/model.py` | `build_model_bin('resnet18')` | fc → Linear(512, 2) |
| `src/dataset.py` | `ScrewDataset`, `train_transform`, `val_transform` | val_transform = Resize(256)+CenterCrop(224)+Normalize |
| `src/utils.py` | `set_seed(42)`, `make_run_dir('./outputs')` | |
| `src/engine.py` | 학습 루프 패턴 **참고** (직접 재구현) | train_with_best_weight 참고 |
| `src/visualization.py` | Grad-CAM target layer 선택 패턴 참고 | `model.layer4[-1]` |

---

## 8. 환경 정보

```
OS: Windows 11
conda env: screw_defect (항상 이 환경에서 실행)
GPU: RTX 2070 SUPER (혹은 5080 — 확인 필요)
CUDA: cu121
torch: 2.5.1+cu121
anomalib: 2.5.0
pytorch_grad_cam: 설치됨
sklearn: 1.8.0
```

실행 전 반드시:
```bash
conda activate screw_defect
cd C:\Users\dlsgh\Desktop\project\screw_defect
```

---

## 9. 커밋 컨벤션

형식: `<타입>: <한 일>` (마침표 X, 명령형)

| 타입 | 용도 |
|------|------|
| `feat` | 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 구조 개선 (동작 변경 없음) |
| `exp` | 실험/결과 기록 |
| `docs` | 문서 |
| `chore` | 설정, gitignore 등 |

트러블슈팅 커밋은 **본문에 원인·해결 기록** — 면접 자산이 됨:
```
fix: Grad-CAM이 no_grad 블록 안에서 동작 안 하는 문제

Grad-CAM은 역전파로 gradient를 계산해야 히트맵이 나옴.
torch.no_grad() 블록 밖에서 호출해야 함.
```

권장 커밋 리듬 (이번 실험):
```
feat: mvtec_data — good 이미지 경로 수집 (train+test 361장)     ← 이미 커밋됨
feat: mvtec_data — 결함 경로+마스크 매핑 및 stratified 분할      ← 다음 커밋
feat: main2.py — ResNet-18 학습 루프 구현
feat: main2.py — 이미지 단위 4지표 (P/R/F2/AUROC) 평가
feat: loc_eval — Grad-CAM pixel AUROC 구현
feat: loc_eval — AUPRO(_AUPRO) 구현
exp: MVTec screw 6지표 최종 결과 및 results_screw_mvtec.csv
```

---

## 10. 작업 방식 (Claude와 협업 방법)

- **코드는 직접 구현** — 트러블슈팅 경험이 포트폴리오/면접 자산
- Claude는 **페어 역할**: 에러 붙여넣으면 원인 진단, 방향 리뷰, API 시그니처 안내
- 막히면 코드/에러 메시지 붙여넣기 → 같이 디버깅
- Claude에게 이 `HANDOVER.md`를 먼저 읽히면 컨텍스트 없이 바로 이어갈 수 있음

---

*최종 업데이트: 2026-07-20 — good 경로 수집까지 완료, 다음: 결함 records + 마스크 매핑*
