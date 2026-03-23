# screw_defect 프로젝트 분석 및 문제점

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **목적** | 나사(screw) 이미지의 불량(bad) / 정상(good) 이진 분류 |
| **데이터** | Train 800장, Val 99장, Test 100장 (ImageFolder: `train/`, `val/`, `test/` 각각 `good/`, `bad/` 하위) |
| **모델** | ResNet18, MobileNetV2, VGG16 (ImageNet pretrained, FC만 2클래스로 교체) |
| **평가** | Bad 클래스 기준 F2-Score, Recall, F1 + Confusion Matrix, 추론 속도(FPS) |
| **특징** | Class weight 스윕(1.0~10.0, 0.5 step)으로 최적 가중치 탐색 후, 해당 가중치로 최종 학습·테스트 |

---

## 2. 아키텍처 및 데이터 흐름

```
main.ipynb
  ├─ set_seed, device, make_run_dir (outputs/run_YYYYMMDD_HHMMSS/)
  ├─ get_downloaders(data_dir, batch_size, num_workers) → dataloaders, image_datasets
  ├─ [모델별] train_and_evaluate_model()
  │    ├─ 가중치 스윕 (19개) × 각각 최대 40 epoch (early stop 가능)
  │    ├─ Val F2 최대인 가중치 선택 → best state_dict 저장
  │    └─ 해당 가중치로 Test 평가 후 반환
  ├─ plot_confusion_matrix, plot_loss_history, plot_weight_tradeoff, show_gradcam_grid
  └─ measure_inference_speed → plot_model_comparison_pareto, plot_model_comparison_bar
```

- **실험 결과 저장**: `outputs/run_*/weights/*.pth`, `outputs/run_*/figures/*.png`
- **지표**: F2 = 5·P·R/(4P+R) (Bad 클래스), E-Score = α·F2 + (1-α)·FPS_norm

---

## 3. 강점

- **재현성**: `set_seed(42)`로 시드 고정.
- **학습 효율**: AMP(혼합 정밀도), Early stopping, DataLoader `persistent_workers`/`prefetch_factor` 적용.
- **평가**: Val F2 기준 최적 class weight 선택 후 Test 1회 평가로 과적합 방지.
- **비교**: 3모델의 F2·FPS·E-Score 시각화(파레토, 막대) 제공.
- **해석**: Grad-CAM 그리드로 Bad/Good·Type1~4별 예측 시각화 시도.

---

## 4. 문제점 및 개선 포인트

### 4.1 버그·잠재 오류

| 위치 | 내용 | 심각도 | 권장 수정 |
|------|------|--------|-----------|
| **utils.py** | `make_run_dir` 내 `now = datetime.now().strftime(...)` 가 동일하게 두 번 반복. | 낮음 | 한 줄 제거. |
| **model.py** | `build_model(model_name)` 에서 `resnet18`/`mobilenet_v2`/`vgg16` 외 이름이 오면 `model`이 정의되지 않아 **NameError** 또는 암묵적 None 반환. | 높음 | `else: raise ValueError(f"Unknown model: {model_name}")` 추가. |
| **visualization.py** | 여러 함수에서 `plt.show()` 후 `plt.savefig()` 호출. 백엔드에 따라 figure가 비워지거나 저장이 잘못될 수 있음. | 중간 | 저장 후 표시 순서로 통일: `plt.savefig(...)` → `plt.show()`. |
| **visualization (Grad-CAM)** | `test/good`, `test/bad` 아래 파일명이 반드시 `good_*.png`, `type1_*.png` 등 prefix를 가진다고 가정. 실제가 `001.png` 등이면 해당 행이 비어 에러 또는 빈 칸. | 중간 | 파일명 조건 완화(예: 해당 폴더 내 모든 이미지 중 샘플) 또는 prefix 옵션/문서화. |

### 4.2 설계·일관성

| 항목 | 내용 | 권장 |
|------|------|------|
| **bad_weights 중복** | `bad_weights = np.arange(1.0, 10.5, 0.5).tolist()` 가 노트북과 `engine.py` 양쪽에 있음. | engine에서만 사용하고, 노트북은 engine 반환값 또는 공통 상수/설정에서 가져오기. |
| **설정 분산** | `num_epochs`, `early_stop_patience`, 가중치 범위/step 등이 노트북·엔진에 흩어져 있음. | `config.py` 또는 단일 dict로 실험 설정 모으기. |
| **DataLoader와 device** | `get_downloaders`는 device를 받지 않음. `pin_memory=True`는 GPU 사용 시에만 의미 있음. | GPU일 때만 `pin_memory=True` 적용하거나, device 인자로 분기. |

### 4.3 데이터·가정

| 항목 | 내용 |
|------|------|
| **클래스 순서** | ImageFolder는 폴더 이름 알파벳 순이므로 `bad=0`, `good=1`. 시각화/엔진의 good=1, bad=0 가정과 일치. |
| **Grad-CAM 데이터 구조** | `test/good/`, `test/bad/` 아래에 `good_*.png`, `type1_*.png`~`type4_*.png` 존재를 전제. 데이터가 다르면 그리드 일부가 비거나 에러. |
| **데이터 크기** | Train 800장은 비교적 작음. Augmentation은 적용 중이나, 과적합 방지를 위해 dropout/weight decay 등 추가 고려 가능. |

### 4.4 성능·품질

| 항목 | 내용 |
|------|------|
| **가중치 스윕 비용** | 19개 가중치 × 최대 40 epoch × 3모델로 학습량이 많음. Early stopping으로 완화됨. |
| **재현성** | `torch.backends.cudnn.deterministic = True` 등 미설정 시 GPU/멀티스레드에서 run 간 미세 차이 가능. |
| **모델 확장** | 새 백본 추가 시 `model.py`의 if-elif 체인 수정 필요. 레지스트리/딕셔너리로 관리하면 확장이 쉬움. |

### 4.5 운영·유지보수

| 항목 | 내용 |
|------|------|
| **의존성** | `requirements.txt` 또는 `pyproject.toml` 없음. torch, torchvision, sklearn, pytorch_grad_cam, matplotlib, seaborn, PIL 등 명시 권장. |
| **에러 처리** | 데이터 경로 없음, 빈 데이터셋, 잘못된 `model_name` 등에 대한 명시적 예외/메시지 부족. |
| **로깅** | `print` 위주. 실험 추적을 위해 로그 파일 또는 간단한 로거 도입 시 유리. |

---

## 5. 요약

- **목적·파이프라인**: 나사 불량 이진 분류와 F2 기반 class weight 탐색, 3모델 비교까지 잘 정의되어 있음.
- **즉시 고치면 좋은 것**: `build_model`에 알 수 없는 `model_name` 시 예외 발생, `utils.py` 중복 줄 제거, figure 저장 순서(`savefig` → `show`) 통일.
- **데이터 가정**: Grad-CAM 그리드는 “test 하위 good/bad + 파일명 prefix” 구조를 가정하므로, 실제 데이터 구조와 맞추거나 코드/문서로 명시하는 것이 좋음.
- **설정·의존성**: 설정 일원화와 `requirements.txt` 추가하면 재현성과 유지보수가 개선됨.

이 문서는 프로젝트 루트의 `PROJECT_ANALYSIS.md`로 저장되어 있으며, 위 권장 사항대로 수정하면 안정성과 확장성이 좋아집니다.
