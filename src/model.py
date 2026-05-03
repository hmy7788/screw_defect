import torch
import torch.nn as nn
import torchvision.models as models

def build_model_bin(model_name):
    """이진 분류(num_classes=2)로 헤드 교체"""
    if model_name == 'resnet18':
        m = models.resnet18(weights='DEFAULT')
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif model_name == 'mobilenet_v2':
        m = models.mobilenet_v2(weights='DEFAULT')
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif model_name == 'vgg16':
        m = models.vgg16(weights='DEFAULT')
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m