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