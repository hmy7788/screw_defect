import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_transforms():
    """Train/Val/Test에 사용할 전처리 기법을 반환합니다."""
    return {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(180), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2), 
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([   # Validation 셋 추가 (Test와 동일한 전처리)
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    }

def get_downloaders(data_dir, batch_size=32, num_workers=2):
    data_transforms = get_transforms()

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    # persistent_workers: 워커 재시작 방지로 epoch 간 속도 유지
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    dataloaders = {}
    for x in ['train', 'val', 'test']:
        loader_kwargs['shuffle'] = (x == 'train')
        dataloaders[x] = DataLoader(image_datasets[x], **loader_kwargs)

    return dataloaders, image_datasets



# main6.ipynb
# =====================================================================
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