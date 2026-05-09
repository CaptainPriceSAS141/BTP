import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes=10):

    model = models.resnet50(weights="DEFAULT")

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model