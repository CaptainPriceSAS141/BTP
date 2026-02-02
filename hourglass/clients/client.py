"""Client-side federated learning logic."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import time


class FederatedClient:
    """Client in split federated learning system.
    
    Paper Section 3.1: Split Federated Learning Workflow
    
    Client responsibilities:
    1. Maintain private local data
    2. Run forward pass on client-side model
    3. Send intermediate features to server
    4. Receive gradients from server
    5. Complete backward pass on client-side model
    6. Update local model via SGD
    """
    
    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.01):
        """Initialize federated client.
        
        Args:
            client_id: Unique client identifier
            model: Client-side model partition
            device: PyTorch device (cpu or cuda)
            learning_rate: SGD learning rate
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer for client-side parameters
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
        
        # Store intermediate features for backward pass
        self.intermediate_features = None
        self.intermediate_features_for_grad = None
    
    def local_train(self,
                   train_loader: DataLoader,
                   num_epochs: int = 1) -> Tuple[List[float], List[float]]:
        """Local training on client device.
        
        This is a placeholder for full local training without server.
        In split FL, training is interleaved with server computation.
        
        Args:
            train_loader: DataLoader with client's local data
            num_epochs: Number of local epochs
            
        Returns:
            (losses, accuracies) for each epoch
        """
        losses = []
        accuracies = []
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass on client
                intermediate_features = self.model(batch_x)
                
                # For now, we don't have full forward pass
                # In actual training, this would be handled by server
                # Here we just store for demonstration
                self.intermediate_features = intermediate_features.detach()
                
            losses.append(epoch_loss)
            accuracies.append(epoch_correct / epoch_total if epoch_total > 0 else 0.0)
        
        return losses, accuracies
    
    def forward_pass(self, batch_x: torch.Tensor) -> torch.Tensor:
        """Run forward pass on client model.
        
        Paper Section 3.1: Client-side forward computation
        
        Args:
            batch_x: Input batch (N, 3, 32, 32)
            
        Returns:
            Intermediate features to send to server
        """
        self.model.eval()
        with torch.no_grad():
            intermediate_features = self.model(batch_x)
        
        return intermediate_features
    
    def backward_pass(self, 
                     gradients_from_server: torch.Tensor,
                     original_input: torch.Tensor) -> None:
        """Run backward pass on client model.
        
        Paper Section 3.1: Client-side backward computation
        
        Args:
            gradients_from_server: Gradients w.r.t. intermediate features
            original_input: Original input batch for recomputation
        """
        self.model.train()
        
        # Recompute intermediate features with gradient tracking
        intermediate_features = self.model(original_input)
        
        # Register hook to backprop gradients
        intermediate_features.backward(gradients_from_server)
        
        # Update client-side model
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters.
        
        Args:
            Returns: Dictionary of parameter tensors
        """
        return {name: param.data.clone().detach()
                for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters (for federated averaging).
        
        Args:
            parameters: Dictionary of parameter tensors
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
    
    def count_parameters(self) -> int:
        """Count total trainable parameters.
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
