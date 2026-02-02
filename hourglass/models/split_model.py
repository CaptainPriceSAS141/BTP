"""Split model utilities for dividing ResNet-18 between client and server."""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


class ClientModel(nn.Module):
    """Client-side model partition (early layers of ResNet-18)."""
    
    def __init__(self, split_point: int = 5):
        """Initialize client model.
        
        Args:
            split_point: Layer index to split at
                        ResNet-18 structure:
                        0: Conv1 + BatchNorm + ReLU + MaxPool
                        1-4: ResNet blocks (layer1-layer4)
                        split_point=5 means up to end of layer1
        """
        super().__init__()
        
        # Load full ResNet-18
        full_model = models.resnet18(pretrained=False)
        
        # Extract early layers
        self.conv1 = full_model.conv1
        self.bn1 = full_model.bn1
        self.relu = full_model.relu
        self.maxpool = full_model.maxpool
        self.layer1 = full_model.layer1
        
        # Only add more layers if split_point > 5
        if split_point > 5:
            self.layer2 = full_model.layer2
        else:
            self.layer2 = None
        
        if split_point > 9:
            self.layer3 = full_model.layer3
        else:
            self.layer3 = None
        
        if split_point > 13:
            self.layer4 = full_model.layer4
        else:
            self.layer4 = None
        
        self.split_point = split_point
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through client model.
        
        Args:
            x: Input tensor (batch_size, 3, 32, 32) for CIFAR-10
            
        Returns:
            Intermediate features to send to server
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks
        x = self.layer1(x)
        
        if self.layer2 is not None:
            x = self.layer2(x)
        if self.layer3 is not None:
            x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        
        return x


class ServerModel(nn.Module):
    """Server-side model partition (deep layers of ResNet-18)."""
    
    def __init__(self, split_point: int = 5, num_classes: int = 10):
        """Initialize server model.
        
        Args:
            split_point: Layer index where client model split
            num_classes: Number of output classes (CIFAR-10: 10)
        """
        super().__init__()
        
        # Load full ResNet-18
        full_model = models.resnet18(pretrained=False)
        
        # Extract deep layers (after split point)
        # For split_point=5, we take from layer2 onwards
        if split_point <= 5:
            self.layer2 = full_model.layer2
            self.layer3 = full_model.layer3
            self.layer4 = full_model.layer4
        elif split_point <= 9:
            self.layer3 = full_model.layer3
            self.layer4 = full_model.layer4
            self.layer2 = None
        elif split_point <= 13:
            self.layer4 = full_model.layer4
            self.layer2 = None
            self.layer3 = None
        else:
            self.layer2 = None
            self.layer3 = None
            self.layer4 = None
        
        # Global average pooling and classification
        self.avgpool = full_model.avgpool
        self.fc = full_model.fc
        
        # Replace final layer for CIFAR-10 if needed
        if num_classes != 1000:
            self.fc = nn.Linear(512, num_classes)
        
        self.split_point = split_point
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through server model.
        
        Args:
            x: Intermediate features from client model
            
        Returns:
            Logits for classification
        """
        # ResNet blocks (conditionally based on split point)
        if hasattr(self, 'layer2') and self.layer2 is not None:
            x = self.layer2(x)
        if hasattr(self, 'layer3') and self.layer3 is not None:
            x = self.layer3(x)
        if hasattr(self, 'layer4') and self.layer4 is not None:
            x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.fc(x)
        
        return x


def get_feature_dimension(split_point: int = 5) -> int:
    """Get the dimension of intermediate features at split point.
    
    Args:
        split_point: Layer index to split at
        
    Returns:
        Feature dimension (channels) at split point
    """
    # ResNet-18 feature dimensions:
    # After conv1 + maxpool: 64 channels
    # After layer1: 64 channels
    # After layer2: 128 channels
    # After layer3: 256 channels
    # After layer4: 512 channels
    
    if split_point <= 5:
        return 64  # After layer1
    elif split_point <= 9:
        return 128  # After layer2
    elif split_point <= 13:
        return 256  # After layer3
    else:
        return 512  # After layer4


def split_model(model: nn.Module, 
                split_layer: int = 5) -> Tuple[nn.Module, nn.Module]:
    """Split a full model into client and server parts.
    
    Args:
        model: Full ResNet-18 model
        split_layer: Layer to split at
        
    Returns:
        Tuple of (client_model, server_model)
    """
    client_model = ClientModel(split_point=split_layer)
    server_model = ServerModel(split_point=split_layer)
    
    return client_model, server_model
