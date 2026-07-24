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
from sklearn.metrics import confusion_matrix, fbeta_score, classification_report
import seaborn as sns

# ==============================
# Grad-CAM 5x10 그리드 시각화
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
        {'label': 'Good',   'dir': f'./data/new_k-fold_data/{base_dir}/good', 'prefix': 'good',  'true_idx': 0},
        {'label': 'Type 1', 'dir': f'./data/new_k-fold_data/{base_dir}/type1',  'prefix': 'type1', 'true_idx': 1},
        {'label': 'Type 2', 'dir': f'./data/new_k-fold_data/{base_dir}/type2',  'prefix': 'type2', 'true_idx': 1},
        {'label': 'Type 3', 'dir': f'./data/new_k-fold_data/{base_dir}/type3',  'prefix': 'type3', 'true_idx': 1},
        {'label': 'Type 4', 'dir': f'./data/new_k-fold_data/{base_dir}/type4',  'prefix': 'type4', 'true_idx': 1},
        {'label': 'Type 5', 'dir': f'./data/new_k-fold_data/{base_dir}/type5',  'prefix': 'type5', 'true_idx': 1},
    ]

    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 14)) 
    fig.suptitle(f'{model_name.upper()} - Prediction vs Grad-CAM', fontsize=20, fontweight='bold', y=1.02)

    for row, config in enumerate(row_configs):
        current_dir = config['dir']

        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        all_files = [f for f in os.listdir(current_dir) if f.lower().endswith(valid_exts)]

        if len(all_files) == 0:
            print(f"[경고] '{current_dir}' 경로에서 이미지를 하나도 찾지 못함")
            continue # 다음 행으로 넘어감
        
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
                    
                pred_text = "Pred: GOOD" if pred_idx == 0 else "Pred: BAD"
                text_color = "lime" if pred_idx == config['true_idx'] else "red"
                
                targets = [ClassifierOutputTarget(config['true_idx'])] 
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


# ==============================
# Grad-CAM 5x10 그리드 시각화 (고정 파일 지정 버전)
# ==============================
def show_gradcam_grid_fixed(model, test_transform, base_dir, device, save_dir, model_name='Model', data_root='./data/new_k-fold_data'):
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
        {'label': 'Good',   'dir': f'{data_root}/{base_dir}/good',  'files': ['325.png', '326.png', '327.png'], 'true_idx': 0},
        {'label': 'Type 1', 'dir': f'{data_root}/{base_dir}/type1', 'files': ['020.png', '021.png', '022.png'], 'true_idx': 1},
        {'label': 'Type 2', 'dir': f'{data_root}/{base_dir}/type2', 'files': ['020.png', '021.png', '022.png'], 'true_idx': 1},
        {'label': 'Type 3', 'dir': f'{data_root}/{base_dir}/type3', 'files': ['020.png', '021.png', '022.png'], 'true_idx': 1},
        {'label': 'Type 4', 'dir': f'{data_root}/{base_dir}/type4', 'files': ['020.png', '021.png', '022.png'], 'true_idx': 1},
        {'label': 'Type 5', 'dir': f'{data_root}/{base_dir}/type5', 'files': ['020.png', '021.png', '022.png'], 'true_idx': 1},
    ]

    fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(10, 14))
    fig.suptitle(f'{model_name.upper()} - Prediction vs Grad-CAM', fontsize=20, fontweight='bold', y=1.02)

    for row, config in enumerate(row_configs):
        current_dir = config['dir']

        for col, file_name in enumerate(config['files']):
            ax = axes[row, col]
            img_path = os.path.join(current_dir, file_name)

            try:
                pil_img = Image.open(img_path).convert('RGB')
                input_tensor = test_transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    pred_idx = torch.argmax(output, dim=1).item()

                pred_text = "Pred: GOOD" if pred_idx == 0 else "Pred: BAD"
                text_color = "lime" if pred_idx == config['true_idx'] else "red"

                targets = [ClassifierOutputTarget(config['true_idx'])]
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
    save_path = os.path.join(save_dir, f'{model_name}_gradcam_fixed_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


#
#
#
def plot_confusion_matrix(y_true, y_pred, class_names, save_dir, model_name="Model"):
    cm = confusion_matrix(y_true, y_pred)
    test_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    test_precision = test_report.get('bad', {}).get('precision', 0.0)
    test_recall = test_report.get('bad', {}).get('recall', 0.0)
    test_f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

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

    print(f'{model_name} Test에 대한 Precision: {test_precision:.4f}')
    print(f'{model_name} Test에 대한 Recall: {test_recall:.4f}')
    print(f'{model_name} Test에 대한 F2-Score: {test_f2:.4f}')

    return test_precision, test_recall, test_f2


def plot_all_models_weight_vs_f2(all_models_results, save_path="./outputs/"):
    """
    모든 모델을 한 그래프에 표시
    
    Args:
        all_models_results = {
            'VGG-16': {
                'weight_f2': {2.0: [0.75, 0.73, ...], 2.5: [...], ...}
            },
            'ResNet-18': {
                'weight_f2': {2.0: [0.79, 0.78, ...], ...}
            },
            'MobileNet-V2': {
                'weight_f2': {2.0: [0.80, 0.81, ...], ...}
            }
        }
    """    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 모델별 색상 & 마커
    model_styles = {
        'VGG-16': {
            'color': '#FF6B6B',  # 빨강
            'marker': 's',       # 사각형
            'linestyle': '--'
        },
        'ResNet-18': {
            'color': '#4ECDC4',  # 청록
            'marker': '^',       # 삼각형
            'linestyle': '-.'
        },
        'MobileNet-V2': {
            'color': '#95E1D3',  # 연두
            'marker': 'o',       # 원
            'linestyle': '-'
        }
    }
    
    best_overall = {'model': None, 'weight': None, 'f2': 0}
    
    for model_name, model_data in all_models_results.items():
        weight_f2 = model_data['weight_f2']
        weights = sorted(weight_f2.keys())
        
        means = [np.mean(weight_f2[w]) for w in weights]
        stds = [np.std(weight_f2[w]) for w in weights]
        
        style = model_styles.get(model_name, {'color': 'gray', 'marker': 'o', 'linestyle': '-'})
        
        # Line + Error bar
        ax.errorbar(
            weights, means, yerr=stds,
            color=style['color'],
            marker=style['marker'],
            markersize=8,
            linestyle=style['linestyle'],
            linewidth=2,
            capsize=5,
            capthick=2,
            label=model_name,
            alpha=0.8
        )
        
        # 이 모델의 best 찾기
        best_idx = np.argmax(means)
        if means[best_idx] > best_overall['f2']:
            best_overall = {
                'model': model_name,
                'weight': weights[best_idx],
                'f2': means[best_idx]
            }
    
    # 전체 best 강조
    if best_overall['model']:
        best_model = best_overall['model']
        best_weight = best_overall['weight']
        best_f2 = best_overall['f2']
        
        ax.scatter(
            best_weight, best_f2,
            marker='*',
            s=600,
            c='gold',
            edgecolors='red',
            linewidths=2,
            zorder=10,
            label=f"Best: {best_model} @ {best_weight:.1f}"
        )
    
    # 축 및 제목
    ax.set_xlabel('Bad Class Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('F2-score', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison: F2-score by Class Weight', 
                fontsize=16, fontweight='bold', pad=20)
    
    # 범례
    ax.legend(fontsize=11, loc='best', framealpha=0.9, shadow=True)
    
    # 그리드
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # X축 눈금
    ax.set_xticks(weights)
    
    # Y축 범위 자동 조정 (여유 공간)
    all_means = []
    all_stds = []
    for model_data in all_models_results.values():
        weight_f2 = model_data['weight_f2']
        for w in weight_f2.keys():
            all_means.append(np.mean(weight_f2[w]))
            all_stds.append(np.std(weight_f2[w]))
    
    y_min = min(all_means) - max(all_stds) - 0.02
    y_max = max(all_means) + max(all_stds) + 0.02
    ax.set_ylim(y_min, y_max)
    
    # 배경색 (선택사항)
    ax.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/all_models_weight_vs_f2.png", 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved: all_models_weight_vs_f2.png")
    print(f"   Best: {best_overall['model']} with F2={best_overall['f2']:.4f} at weight={best_overall['weight']:.1f}")