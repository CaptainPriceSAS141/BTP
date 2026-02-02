"""Server-side model partition."""

import torch
import torch.nn as nn
from torchvision import models


class ServerModelPartition(nn.Module):
    """Server-side model partition (deep layers of ResNet-18).
    
    Paper Section 3.2: Hourglass Server Design
    - ONE shared model per GPU (not per client)
    - Processes intermediate features from multiple clients
    - Returns gradients back to clients
    """
    
    def __init__(self, split_layer: int = 5, num_classes: int = 10):
        """Initialize server model.
        
        Args:
            split_layer: ResNet-18 layer to split at
            num_classes: Number of output classes (CIFAR-10: 10)
        """
        super().__init__()
        
        # Load base ResNet-18
        base_model = models.resnet18(pretrained=False)
        
        # Extract deep layers (after split)
        # For split_layer=5 (after layer1), include layer2, layer3, layer4
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        
        # Global pooling and classification head
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(512, num_classes)
        
        self.split_layer = split_layer
        self.num_classes = num_classes
    
    def forward(self, intermediate_features: torch.Tensor) -> torch.Tensor:
        """Forward pass on server.
        
        Args:
            intermediate_features: Features from client model
                                  Shape: (N, 64, 8, 8) for CIFAR-10
            
        Returns:
            Logits for classification (N, num_classes)
        """
        x = self.layer2(intermediate_features)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_intermediate_gradient_hook(self):
        """Get hook to capture gradients w.r.t. intermediate features.
        
        Used to send gradients back to clients for their backward pass.
        
        Returns:
            Hook function
        """
        def hook(grad):
            # Store gradient for transmission to client
            self._intermediate_gradient = grad
            return grad
        
        return hook
