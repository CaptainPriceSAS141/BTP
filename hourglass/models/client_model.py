"""Client-side model partition."""

import torch
import torch.nn as nn
from torchvision import models


class ClientModelPartition(nn.Module):
    """Client-side model partition (early layers of ResNet-18).
    
    Paper Section 3.1: Client-side model partition
    Runs on client devices, produces intermediate features.
    """
    
    def __init__(self, split_layer: int = 5):
        """Initialize client model.
        
        Args:
            split_layer: ResNet-18 layer index to split at
        """
        super().__init__()
        
        # Load pretrained ResNet-18 (or initialize random)
        base_model = models.resnet18(pretrained=False)
        
        # Extract early layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        
        self.split_layer = split_layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on client device.
        
        Args:
            x: Input batch (N, 3, 32, 32)
            
        Returns:
            Intermediate features (N, 64, 8, 8) for CIFAR-10 split at layer1
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        
        return x
