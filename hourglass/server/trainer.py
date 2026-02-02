"""Server-side training logic for split federated learning.

Paper Section 3.2: Server-side Training
Implements forward pass on intermediate features and gradient computation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
import time


class ServerTrainer:
    """Server trainer for split federated learning.
    
    Paper Section 3.2: Hourglass Server Design
    - Maintains ONE shared server-side model per GPU
    - Processes intermediate features from multiple clients
    - Computes loss and backward pass
    - Returns gradients to clients
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.01):
        """Initialize server trainer.
        
        Args:
            model: Server-side model partition
            device: PyTorch device
            learning_rate: Learning rate for server-side optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer for server-side parameters
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.total_loss = 0.0
        self.total_samples = 0
    
    def forward_pass(self, 
                    intermediate_features: torch.Tensor) -> torch.Tensor:
        """Run forward pass on server.
        
        Args:
            intermediate_features: Features from client model
            
        Returns:
            Logits for classification
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(intermediate_features)
        
        return logits
    
    def backward_pass(self,
                     intermediate_features: torch.Tensor,
                     targets: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """Backward pass on server.
        
        Computes loss, gradients, and returns gradients w.r.t. intermediate features.
        
        Args:
            intermediate_features: Features from client model
                                  (requires_grad=True for gradient flow)
            targets: Ground truth labels
            
        Returns:
            (gradients_wrt_intermediate_features, loss, num_samples)
        """
        # Ensure we track gradients for intermediate features
        if not intermediate_features.requires_grad:
            intermediate_features = intermediate_features.requires_grad_(True)
        
        # Forward pass
        logits = self.model(intermediate_features)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Get gradients w.r.t. intermediate features
        gradients = intermediate_features.grad.clone()
        
        # Update server-side parameters
        self.optimizer.step()
        
        # Update metrics
        self.total_loss += loss.item() * targets.size(0)
        self.total_samples += targets.size(0)
        
        return gradients, loss.item(), targets.size(0)
    
    def forward_backward_pass(self,
                             intermediate_features: torch.Tensor,
                             targets: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Combined forward-backward pass.
        
        Args:
            intermediate_features: Features from client model
            targets: Ground truth labels
            
        Returns:
            (gradients_wrt_intermediate_features, loss_value)
        """
        # Enable gradient computation for intermediate features
        intermediate_features = intermediate_features.detach()
        intermediate_features.requires_grad = True
        
        # Forward pass
        logits = self.model(intermediate_features)
        
        # Compute loss
        loss = self.criterion(logits, targets)
        
        # Backward pass for loss
        self.optimizer.zero_grad()
        loss.backward()
        
        # Extract gradients w.r.t. intermediate features
        if intermediate_features.grad is not None:
            gradients = intermediate_features.grad.clone().detach()
        else:
            # If no gradients computed, return zeros
            gradients = torch.zeros_like(intermediate_features)
        
        # Update server parameters
        self.optimizer.step()
        
        return gradients, loss.item()
    
    def get_loss(self) -> float:
        """Get average loss for current batch/epoch.
        
        Returns:
            Average loss
        """
        if self.total_samples == 0:
            return 0.0
        
        return self.total_loss / self.total_samples
    
    def reset_metrics(self):
        """Reset loss metrics."""
        self.total_loss = 0.0
        self.total_samples = 0
    
    def get_parameters(self) -> dict:
        """Get current model parameters.
        
        Returns:
            Dictionary of parameter tensors
        """
        return {name: param.data.clone().detach()
                for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: dict) -> None:
        """Set model parameters.
        
        Args:
            parameters: Dictionary of parameter tensors
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
    
    def count_parameters(self) -> int:
        """Count trainable parameters.
        
        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
