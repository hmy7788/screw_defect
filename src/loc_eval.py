import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from anomalib.metrics.aupro import _AUPRO

from src.dataset import val_transform

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

mask_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(224),
])


def load_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask = mask_transform(mask)
    mask = np.array(mask, dtype=np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    return mask


def _collect_cam_and_mask(model, defect_records, device):
    """defect_records 각각에 대해 Grad-CAM 히트맵과 GT mask를 계산해 (type, cam, mask) 리스트로 반환."""
    cam_extractor = GradCAM(model=model, target_layers=[model.layer4[-1]])
    targets = [ClassifierOutputTarget(1)]

    model.eval()
    results = []
    for r in defect_records:
        img = Image.open(r['path']).convert('RGB')
        input_tensor = val_transform(img).unsqueeze(0).to(device)

        # no_grad 밖에서 호출 (Grad-CAM은 역전파로 heatmap을 계산함)
        grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]  # (224, 224)
        mask = load_mask(r['mask'])  # (224, 224)

        results.append((r['type'], grayscale_cam, mask))

    return results


def _compute_pixel_metrics(cam_list, mask_list):
    cam_arr = np.stack(cam_list)    # (N, 224, 224)
    mask_arr = np.stack(mask_list)  # (N, 224, 224)

    pixel_auroc = roc_auc_score(mask_arr.reshape(-1), cam_arr.reshape(-1))

    cam_tensor = torch.from_numpy(cam_arr).float()    # (N, 224, 224)
    mask_tensor = torch.from_numpy(mask_arr).float()  # (N, 224, 224)

    aupro_metric = _AUPRO()
    aupro_metric.update(cam_tensor, mask_tensor)
    aupro_score = aupro_metric.compute().item()

    return {'pixel_auroc': pixel_auroc, 'aupro': aupro_score}


def evaluate_localization(model, test_records, device):
    """test_records 중 결함(label==1) 이미지만 대상으로 CAM과 GT mask를 비교해
    Pixel AUROC / AUPRO를 계산한다."""
    defect_records = [r for r in test_records if r['label'] == 1]

    collected = _collect_cam_and_mask(model, defect_records, device)
    cam_list = [c for _, c, _ in collected]
    mask_list = [m for _, _, m in collected]

    return _compute_pixel_metrics(cam_list, mask_list)


def evaluate_localization_by_type(model, test_records, device):
    """결함 유형(type)별로 Pixel AUROC / AUPRO를 따로 계산해 dict로 반환."""
    defect_records = [r for r in test_records if r['label'] == 1]

    collected = _collect_cam_and_mask(model, defect_records, device)

    by_type = {}
    for t, cam, mask in collected:
        by_type.setdefault(t, {'cam': [], 'mask': []})
        by_type[t]['cam'].append(cam)
        by_type[t]['mask'].append(mask)

    results = {}
    for t, data in by_type.items():
        results[t] = _compute_pixel_metrics(data['cam'], data['mask'])
        results[t]['n'] = len(data['cam'])

    return results


def show_localization_grid(model, test_records, device, save_dir='./outputs', model_name='ResNet-18'):
    """정상(good) 및 결함 유형별로 한 장씩 뽑아 원본 / Grad-CAM 오버레이 / GT 마스크를 나란히 시각화."""
    all_types = sorted({r['type'] for r in test_records})
    # good을 맨 위에 오도록 정렬
    all_types = ['good'] + [t for t in all_types if t != 'good'] if 'good' in all_types else all_types

    by_type = {}
    for r in test_records:
        by_type.setdefault(r['type'], []).append(r)

    samples = []
    for t in all_types:
        if t not in by_type or len(by_type[t]) == 0:
            print(f"[경고] '{t}' 타입의 test 샘플이 없어 grid에서 제외됨")
            continue
        samples.append(random.choice(by_type[t]))

    cam_extractor = GradCAM(model=model, target_layers=[model.layer4[-1]])

    model.eval()
    n_rows = len(samples)
    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ['Original', 'Grad-CAM', 'GT Mask']

    for row, r in enumerate(samples):
        img = Image.open(r['path']).convert('RGB')
        input_tensor = val_transform(img).unsqueeze(0).to(device)

        target_idx = r['label']  # good=0, bad=1
        targets = [ClassifierOutputTarget(target_idx)]

        # no_grad 밖에서 호출 (Grad-CAM은 역전파로 heatmap을 계산함)
        grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]

        inv_tensor = inv_normalize(input_tensor.squeeze(0))
        vis_img_np = np.clip(inv_tensor.cpu().numpy().transpose((1, 2, 0)), 0, 1)
        cam_overlay = show_cam_on_image(vis_img_np, grayscale_cam, use_rgb=True)

        mask = load_mask(r['mask']) if r['mask'] is not None else np.zeros((224, 224), dtype=np.float32)

        axes[row, 0].imshow(vis_img_np)
        axes[row, 1].imshow(cam_overlay)
        axes[row, 2].imshow(mask, cmap='gray', vmin=0, vmax=1)

        for col in range(3):
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=13, fontweight='bold')

        axes[row, 0].text(-0.15, 0.5, r['type'], fontsize=11, fontweight='bold', color='darkred',
                           va='center', ha='right', rotation='vertical', transform=axes[row, 0].transAxes)

    fig.suptitle(f'{model_name} - Grad-CAM vs GT Mask', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_localization_grid.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
