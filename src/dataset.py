import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class ScrewDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]
    

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(180, fill=(196, 196, 196)),   # /new_k-fold_data/train/good/000.png의 가장 왼쪽위 픽셀값: 196, 196, 196
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_image_paths_and_labels(DATA_DIR, CLASS_NAMES):
    # 이진 라벨로 통일: good=0, type1~type5=1(bad)
    all_paths, all_labels = [], []
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"클래스 폴더 없음: {class_dir}")

        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(valid_exts):
                all_paths.append(os.path.join(class_dir, file_name))
                all_labels.append(class_idx)

    all_labels_np = np.array(all_labels, dtype=np.int64)

    # 분포 확인
    print(f"전체 샘플: {len(all_paths)}장")
    for idx, name in enumerate(CLASS_NAMES):
        print(f'{name}({idx}): {(all_labels_np == idx).sum()}장')

    if len(all_paths) == 0:
        raise RuntimeError("로드된 이미지가 0장입니다. 확장자/경로를 확인하세요.")
    
    return all_paths, all_labels_np