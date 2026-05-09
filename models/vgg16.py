from typing import cast

import torch.nn as nn
from torchvision import models

def get_vgg16(num_classes=10):

    model = models.vgg16(weights="DEFAULT")

    last_layer = cast(nn.Linear, model.classifier[6])
    model.classifier[6] = nn.Linear(
        last_layer.in_features,
        num_classes
    )

    return model