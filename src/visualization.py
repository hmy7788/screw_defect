import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ========================================================
# 1. 학습 로스 추이 그래프 (에포크 / train loss, val loss)
# ========================================================
def plot_loss_history(train_loss, val_loss, save_dir, model_name="Model"):
    """
    최고 성능을 낸 모델의 에포크별 loss 추이 그래프
    """
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='', linestyle='-', label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, marker='', linestyle='-', label='Val Loss', color='orange', linewidth=2)

    plt.title(f'{model_name.upper()} Loss Curve (Best Model)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# 2. 가중치 스윕 트레이드 오프 그래프 (bad weight / recall, precision, f1, f2)
# ============================================================================
def plot_weight_tradeoff(weights, recalls, f1s, f2s, save_dir, model_name='Model'):
    """
    가중치 변화에 따른 recall, f1, f2 변화
    weights: bad class weight 리스트 (예: np.arange(1.0, 10.5, 0.5).tolist())
    """
    weights = list(weights)
    plt.figure(figsize=(10, 6))
    plt.plot(weights, recalls, marker='o', label='Recall', color='red')
    plt.plot(weights, f1s, marker='o', label='F1-Score', color='green')
    plt.plot(weights, f2s, marker='o', label='F2-Score', color='blue')

    plt.title(f'{model_name.upper()} - Metric by Class Weight', fontsize=16, fontweight='bold')
    plt.xlabel('Bad Class Weight', fontsize=12)
    plt.ylabel('Score (0.0 ~ 1.0)', fontsize=12)
    plt.xticks(weights)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    max_f2_idx = np.argmax(f2s)
    optimal_weight = weights[max_f2_idx]
    max_f2 = f2s[max_f2_idx]

    plt.annotate(f'Optimal Weight: {optimal_weight}\n Max F2 : {max_f2:.4f}', 
                 xy=(optimal_weight, max_f2),
                 xytext=(optimal_weight, max_f2 + 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=12, fontweight='bold', ha='center')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_weight_tradeoff.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ==================
# 3. 파레토 프론티어 
# ==================
def plot_model_comparison_pareto(model_names, f2_scores, fps_list, alpha, save_dir, device_label=None):
    """
    3개 모델의 FPS와 F2-Score를 바탕으로 E-Score(가중 산술 평균)를 시각화
    device_label: 'CPU' 또는 'GPU' 등이면 제목·파일명에 반영 (예: _cpu.png)
    """
    n = len(model_names)
    # 비교 셀을 여러 번 실행하면 리스트가 쌓이므로, 항상 마지막 n개만 사용
    f2_scores = np.asarray(f2_scores, dtype=float).flatten()[-n:]
    fps_list = np.asarray(fps_list, dtype=float).flatten()[-n:]
    if len(f2_scores) != n or len(fps_list) != n:
        raise ValueError(f"f2_scores와 fps_list는 model_names 개수({n})와 같아야 합니다. 노트북에서 '초기화 셀' 실행 후 모델 3개 학습 셀을 각각 한 번만 실행하세요.")
    # 정규화 (최대 FPS 기준)
    max_fps = float(np.max(fps_list))
    fps_norm = (fps_list / max_fps).tolist()
    
    # E-Score 계산 (알파 값 적용)
    e_scores = [ (alpha * f2) + ((1 - alpha) * fn) for f2, fn in zip(f2_scores, fps_norm) ]
    
    plt.figure(figsize=(9, 6))
    colors = ['purple', 'orange', 'teal']
    
    for i in range(len(model_names)):
        plt.scatter(fps_list[i], f2_scores[i], color=colors[i], s=200, label=f"{model_names[i].upper()} (E-Score: {e_scores[i]:.3f})")
        plt.text(fps_list[i], f2_scores[i] + 0.005, model_names[i].upper(), fontsize=12, ha='center', fontweight='bold')
    
    title_suffix = f' ({device_label})' if device_label else ''
    plt.title(f'Pareto Frontier: FPS vs F2-Score (α={alpha}){title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel('Inference Speed (FPS)', fontsize=12)
    plt.ylabel('F2-Score', fontsize=12)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    name_suffix = f'_{device_label.lower()}' if device_label else ''
    save_path = os.path.join(save_dir, f'model_pareto_comparison{name_suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_model_comparison_bar(model_names, f2_scores, fps_list, alpha, save_dir, device_label=None):
    """
    3개 모델의 F2-Score, 정규화된 FPS, 종합 E-Score를 그룹 막대 그래프로 비교
    device_label: 'CPU' 또는 'GPU' 등이면 제목·파일명에 반영
    """
    n = len(model_names)
    # 비교 셀을 여러 번 실행하면 리스트가 쌓이므로, 항상 마지막 n개만 사용
    f2_scores = np.asarray(f2_scores, dtype=float).flatten()[-n:]
    fps_list = np.asarray(fps_list, dtype=float).flatten()[-n:]
    if len(f2_scores) != n or len(fps_list) != n:
        raise ValueError(f"f2_scores와 fps_list는 model_names 개수({n})와 같아야 합니다. 현재 len(f2_scores)={len(f2_scores)}, len(fps_list)={len(fps_list)}. 노트북에서 '초기화 셀'을 실행한 뒤 모델 3개 학습 셀을 각각 한 번만 실행하세요.")
    # 1. 수치 계산 (정규화 및 E-Score)
    max_fps = float(np.max(fps_list))
    fps_norm = (fps_list / max_fps).tolist()

    # E-Score: F2와 FPS_norm의 가중 산술 평균 (알파 값 적용)
    e_scores = [ (alpha * f2) + ((1 - alpha) * fn) for f2, fn in zip(f2_scores, fps_norm) ]

    # 2. 막대 그래프 세팅
    x = np.arange(n)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. n개의 막대 그리기 (길이 n인 리스트만 전달)
    f2_plot = f2_scores.tolist() if hasattr(f2_scores, 'tolist') else list(f2_scores)
    fps_norm_plot = fps_norm if isinstance(fps_norm, list) else list(fps_norm)
    e_scores_plot = list(e_scores)
    rects1 = ax.bar(x - width, f2_plot, width, label='F2-Score', color='royalblue')
    rects2 = ax.bar(x, fps_norm_plot, width, label='FPS', color='mediumseagreen')
    rects3 = ax.bar(x + width, e_scores_plot, width, label=f'E-Score (α={alpha})', color='darkorange')

    # 4. 그래프 꾸미기
    title_suffix = f' ({device_label})' if device_label else ''
    ax.set_ylabel('Scores (0.0 ~ 1.0 Scale)', fontsize=12)
    ax.set_title(f'Model Performance Comparison (F2, FPS, E-Score){title_suffix}', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.upper() for name in model_names], fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11) # 범례를 그래프 바깥으로 빼서 가리지 않게 함

    # y축 범위를 0 ~ 1.2 정도로 살짝 늘려서 텍스트가 잘리지 않게 함
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. 막대 위에 수치 적어주기
    def autolabel(rects, real_values, is_fps=False):
        for rect, val in zip(rects, real_values):
            height = rect.get_height()
            # FPS면 '45 FPS' 처럼 쓰고, 아니면 '0.952' 처럼 소수점 3자리로 씀
            text_str = f"{val:.1f} FPS" if is_fps else f"{val:.3f}"
            ax.annotate(text_str,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 막대 위로 3포인트 띄움
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1, f2_plot)
    autolabel(rects2, (fps_list.tolist() if hasattr(fps_list, 'tolist') else list(fps_list)), is_fps=True)
    autolabel(rects3, e_scores_plot)

    plt.tight_layout()
    name_suffix = f'_{device_label.lower()}' if device_label else ''
    save_path = os.path.join(save_dir, f'model_comparison_barchart{name_suffix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ==============================
# 4. Grad-CAM 5x10 그리드 시각화
# ==============================
def show_gradcam_grid(model, test_transform, base_dir, device, save_dir, model_name='Model'):
    """
    """
    model.eval()

    if 'resnet' in model_name.lower():
        target_layers = [model.layer4[-1]]
    elif 'mobilenet' in model_name.lower():
        target_layers = [model.features[-1]]
    elif 'vgg' in model_name.lower():
        target_layers = [model.features[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    row_configs = [
        {'label': 'Good',   'dir': f'{base_dir}/good', 'prefix': 'good',  'true_idx': 0},
        {'label': 'Type 1', 'dir': f'{base_dir}/type1',  'prefix': 'type1', 'true_idx': 1},
        {'label': 'Type 2', 'dir': f'{base_dir}/type2',  'prefix': 'type2', 'true_idx': 2},
        {'label': 'Type 3', 'dir': f'{base_dir}/type3',  'prefix': 'type3', 'true_idx': 3},
        {'label': 'Type 4', 'dir': f'{base_dir}/type4',  'prefix': 'type4', 'true_idx': 4},
        {'label': 'Type 5', 'dir': f'{base_dir}/type5',  'prefix': 'type5', 'true_idx': 5},
    ]

    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10, 14)) 
    fig.suptitle(f'{model_name.upper()} - Prediction vs Grad-CAM', fontsize=20, fontweight='bold', y=1.02)

    for row, config in enumerate(row_configs):
        current_dir = config['dir']
        all_files = [f for f in os.listdir(current_dir) if f.startswith(config['prefix']) and f.endswith('.png')]
        
        # 사진 3장 랜덤 추출
        sample_size = min(3, len(all_files))
        random_files = random.sample(all_files, sample_size)
        
        for col, file_name in enumerate(random_files):
            ax = axes[row, col]
            img_path = os.path.join(current_dir, file_name)
            
            try:
                pil_img = Image.open(img_path).convert('RGB')
                input_tensor = test_transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()
                    
                pred_text = "Pred: GOOD" if pred_idx == 1 else "Pred: BAD"
                text_color = "lime" if pred_idx == config['true_idx'] else "red"
                
                targets = [ClassifierOutputTarget(0)] 
                inv_tensor = inv_normalize(input_tensor.squeeze(0))
                vis_img_np = np.clip(inv_tensor.cpu().numpy().transpose((1, 2, 0)), 0, 1)

                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
                visualization = show_cam_on_image(vis_img_np, grayscale_cam, use_rgb=True)
                
                ax.imshow(visualization)
                ax.set_title(pred_text, color=text_color, fontweight='bold', fontsize=12, backgroundcolor='black')
                ax.text(0.5, -0.05, file_name, fontsize=10, color='dimgray', ha='center', va='top', transform=ax.transAxes)
                
            except Exception:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', color='gray')
                
            ax.axis('off')
            
            # 행렬 첫 번째 열에만 타입 라벨 적어주기
            if col == 0:
                ax.text(-0.15, 0.5, config['label'], fontsize=16, fontweight='bold', color='darkblue' if row==0 else 'darkred',
                        va='center', ha='right', rotation='vertical', transform=ax.transAxes)

    plt.tight_layout(h_pad=1.5, w_pad=1.0)
    save_path = os.path.join(save_dir, f'{model_name}_gradcam_paper_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 
# 
# 
def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5)) # 그래프 세팅

    # seaborn 히트맵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"}, cbar=False)
    
    # 축과 타이틀 꾸미기
    plt.title(f'{model_name.upper()} - Confusion Matrix', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()