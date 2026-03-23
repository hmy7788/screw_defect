import torch
import torch.nn as nn
import torchvision.models as models

def build_model(model_name, num_classes=2):
    """
    """
    if model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='DEFAULT')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model_name == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose among: resnet18, mobilenet_v2, vgg16")

    return model

def load_best_weights(model, weights_path, device):
    """
    """
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model