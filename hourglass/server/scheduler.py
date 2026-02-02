"""Scheduling strategies for split federated learning.

Paper Section 4: Hourglass Scheduler
Implements FCFS and DFF scheduling to decide which GPU processes incoming client requests.
"""

import numpy as np
from typing import List, Tuple, Dict
import torch


class Scheduler:
    """Base scheduler for client request processing."""
    
    def __init__(self, num_gpus: int = 1):
        """Initialize scheduler.
        
        Args:
            num_gpus: Number of available GPUs
        """
        self.num_gpus = num_gpus
        self.request_queue = []
    
    def add_request(self, client_id: int, features: torch.Tensor) -> None:
        """Add client request to queue.
        
        Args:
            client_id: Client identifier
            features: Intermediate features from client
        """
        self.request_queue.append({
            'client_id': client_id,
            'features': features,
            'timestamp': len(self.request_queue)
        })
    
    def get_next_gpu(self) -> Tuple[int, Dict]:
        """Get next GPU and associated request.
        
        Returns:
            (gpu_id, request_dict)
        """
        raise NotImplementedError


class FCFSScheduler(Scheduler):
    """First Come First Serve scheduler.
    
    Paper Section 4.1: FCFS Strategy
    Simplest scheduling: process requests in arrival order.
    Assigns incoming requests to least-loaded GPU.
    """
    
    def __init__(self, num_gpus: int = 1):
        """Initialize FCFS scheduler.
        
        Args:
            num_gpus: Number of available GPUs
        """
        super().__init__(num_gpus)
        self.gpu_load = [0] * num_gpus  # Track requests per GPU
    
    def add_request(self, client_id: int, features: torch.Tensor) -> None:
        """Add request and assign to least-loaded GPU."""
        super().add_request(client_id, features)
        
        # Assign to least-loaded GPU
        assigned_gpu = np.argmin(self.gpu_load)
        self.request_queue[-1]['gpu_id'] = assigned_gpu
        self.gpu_load[assigned_gpu] += 1
    
    def get_next_gpu(self) -> Tuple[int, Dict]:
        """Get next request in FCFS order.
        
        Returns:
            (gpu_id, request)
        """
        if not self.request_queue:
            return None, None
        
        request = self.request_queue.pop(0)
        gpu_id = request['gpu_id']
        self.gpu_load[gpu_id] -= 1
        
        return gpu_id, request
    
    def reset(self):
        """Reset scheduler state."""
        self.request_queue = []
        self.gpu_load = [0] * self.num_gpus


class DFFScheduler(Scheduler):
    """Dissimilar Feature First (DFF) scheduler.
    
    Paper Section 4.1: DFF Strategy
    Orders client requests by feature dissimilarity to reduce GPU cache misses.
    Features that are dissimilar are processed consecutively.
    """
    
    def __init__(self, num_gpus: int = 1, embedding_dim: int = 128):
        """Initialize DFF scheduler.
        
        Args:
            num_gpus: Number of available GPUs
            embedding_dim: Dimension of feature embeddings for similarity
        """
        super().__init__(num_gpus)
        self.gpu_load = [0] * num_gpus
        self.embedding_dim = embedding_dim
        self.embeddings = []  # Store feature embeddings
    
    def _compute_embedding(self, features: torch.Tensor) -> np.ndarray:
        """Compute embedding for features.
        
        Uses average pooling to create a compact representation.
        
        Args:
            features: Intermediate features (N, C, H, W)
            
        Returns:
            Embedding vector (embedding_dim,)
        """
        # Global average pooling
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        pooled = pooled.view(pooled.size(0), -1)  # (N, C)
        
        # Take mean across batch
        embedding = pooled.mean(dim=0).cpu().detach().numpy()
        
        # Project to embedding_dim using simple PCA-like approach
        if embedding.shape[0] > self.embedding_dim:
            # Truncate if too large
            embedding = embedding[:self.embedding_dim]
        else:
            # Pad if too small
            embedding = np.pad(embedding, 
                             (0, self.embedding_dim - embedding.shape[0]))
        
        return embedding
    
    def _compute_dissimilarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute dissimilarity between two embeddings.
        
        Uses Euclidean distance.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Dissimilarity score (higher = more dissimilar)
        """
        return np.linalg.norm(emb1 - emb2)
    
    def add_request(self, client_id: int, features: torch.Tensor) -> None:
        """Add request and compute embedding."""
        super().add_request(client_id, features)
        
        # Compute embedding
        embedding = self._compute_embedding(features)
        self.embeddings.append(embedding)
    
    def get_next_gpu(self) -> Tuple[int, Dict]:
        """Get next request using DFF strategy.
        
        Returns the request that is most dissimilar to recently processed requests.
        """
        if not self.request_queue:
            return None, None
        
        # If this is the first request, just pick the least-loaded GPU
        if not hasattr(self, '_last_processed_embedding'):
            self._last_processed_embedding = None
        
        best_idx = 0
        best_dissimilarity = -1
        
        # Find most dissimilar request
        for i, request in enumerate(self.request_queue):
            embedding = self.embeddings[len(self.embeddings) - len(self.request_queue) + i]
            
            if self._last_processed_embedding is None:
                dissimilarity = 0
            else:
                dissimilarity = self._compute_dissimilarity(
                    embedding, self._last_processed_embedding
                )
            
            if dissimilarity > best_dissimilarity:
                best_dissimilarity = dissimilarity
                best_idx = i
        
        # Get selected request
        request = self.request_queue.pop(best_idx)
        self._last_processed_embedding = self.embeddings.pop(best_idx)
        
        # Assign to least-loaded GPU
        assigned_gpu = np.argmin(self.gpu_load)
        request['gpu_id'] = assigned_gpu
        self.gpu_load[assigned_gpu] += 1
        
        return assigned_gpu, request
    
    def reset(self):
        """Reset scheduler state."""
        super().reset()
        self.gpu_load = [0] * self.num_gpus
        self.embeddings = []
        self._last_processed_embedding = None


def create_scheduler(scheduler_type: str = "fcfs", num_gpus: int = 1) -> Scheduler:
    """Factory function to create scheduler.
    
    Args:
        scheduler_type: "fcfs" or "dff"
        num_gpus: Number of available GPUs
        
    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == "fcfs":
        return FCFSScheduler(num_gpus)
    elif scheduler_type.lower() == "dff":
        return DFFScheduler(num_gpus)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
