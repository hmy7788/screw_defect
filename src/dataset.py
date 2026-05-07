import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import StratifiedKFold

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


def full_train_test_split(all_paths, seed):
    # 1. 파라미터로 받은 seed 값 적용 (고정된 42 대신 변수 사용)
    random.seed(seed)
    
    CLASS_NAMES = ["good", "type1", "type2", "type3", "type4", "type5"]
    
    # 민엽님이 원하시는 정확한 테스트 셋 개수
    TARGET_TEST_COUNTS = {
        'good': 36, 'type1': 4, 'type2': 4, 
        'type3': 5, 'type4': 3, 'type5': 3
    }
    
    # 2. 클래스별로 파일 경로 분류하기
    paths_by_class = {c: [] for c in CLASS_NAMES}
    for p in all_paths:
        for c in CLASS_NAMES:
            if c in p:
                paths_by_class[c].append(p)
                break
                
    train_paths, test_paths = [], []
    train_labels, test_labels = [], []
    
    # 3. 각 클래스별로 리스트를 섞고, 원하는 개수만큼 정확히 자르기
    for class_name in CLASS_NAMES:
        paths = paths_by_class[class_name].copy()
        random.shuffle(paths) # 시드에 맞춰 무작위로 섞기
        
        n_test = TARGET_TEST_COUNTS[class_name]
        
        test_p = paths[:n_test]
        train_p = paths[n_test:]
        
        test_paths.extend(test_p)
        train_paths.extend(train_p)
        
        # 라벨링 (good은 0, 나머지는 1)
        label = 0 if class_name == 'good' else 1
        test_labels.extend([label] * len(test_p))
        train_labels.extend([label] * len(train_p))
        
    # 기존 sklearn의 train_test_split과 똑같이 4개의 결과값을 반환
    return train_paths, test_paths, np.array(train_labels), np.array(test_labels)


def custom_Stratified_K_Fold(full_train_paths, n_splits=5, seed=42):
    train_detailed_labels = []
    for p in full_train_paths:
        if 'good' in p: train_detailed_labels.append('good')
        elif 'type1' in p: train_detailed_labels.append('type1')
        elif 'type2' in p: train_detailed_labels.append('type2')
        elif 'type3' in p: train_detailed_labels.append('type3')
        elif 'type4' in p: train_detailed_labels.append('type4')
        elif 'type5' in p: train_detailed_labels.append('type5')

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_splits = list(skf.split(full_train_paths, train_detailed_labels))
    
    return fold_splits, train_detailed_labels