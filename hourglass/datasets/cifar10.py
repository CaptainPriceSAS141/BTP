"""CIFAR-10 dataset loading with non-IID data partitioning."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict


class CIFAR10NonIID:
    """CIFAR-10 dataset with non-IID partitioning across clients."""
    
    def __init__(self, data_dir: str = "./data", seed: int = 42):
        """Initialize CIFAR-10 dataset.
        
        Args:
            data_dir: Directory to store/load dataset
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # No augmentation for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        self.train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        self.test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
    
    def create_non_iid_split(self, 
                            num_clients: int, 
                            alpha: float = 0.1) -> Dict[int, List[int]]:
        """Create non-IID data split using Dirichlet distribution.
        
        Based on:
        "Federated Learning with Non-IID Data"
        https://arxiv.org/abs/1909.06335
        
        Args:
            num_clients: Number of federated clients
            alpha: Dirichlet distribution parameter
                  - alpha=0 -> completely non-IID
                  - alpha=infinity -> IID
                  - alpha=0.1 -> highly non-IID (default)
        
        Returns:
            Dictionary mapping client_id -> list of sample indices
        """
        num_classes = 10  # CIFAR-10
        labels = np.array(self.train_dataset.targets)
        
        # Create class-to-indices mapping
        class_indices = [np.where(labels == c)[0] for c in range(num_classes)]
        
        # Use Dirichlet distribution to allocate classes to clients
        client_data = {i: [] for i in range(num_clients)}
        
        for c in range(num_classes):
            # Sample from Dirichlet
            proportions = np.random.dirichlet([alpha] * num_clients)
            
            # Distribute class samples to clients
            indices = class_indices[c]
            np.random.shuffle(indices)
            
            split_indices = (np.cumsum(proportions) * len(indices)).astype(int)
            splits = np.split(indices, split_indices)[:-1]
            
            for client_id, client_indices in enumerate(splits):
                client_data[client_id].extend(client_indices)
        
        # Shuffle within each client
        for client_id in client_data:
            np.random.shuffle(client_data[client_id])
        
        return client_data
    
    def get_client_train_loader(self,
                               client_id: int,
                               client_data: Dict[int, List[int]],
                               batch_size: int = 32) -> DataLoader:
        """Get DataLoader for a specific client.
        
        Args:
            client_id: Client ID
            client_data: Dictionary from create_non_iid_split()
            batch_size: Batch size
            
        Returns:
            DataLoader for client's training data
        """
        indices = client_data[client_id]
        subset = Subset(self.train_dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True)
    
    def get_test_loader(self, batch_size: int = 32) -> DataLoader:
        """Get DataLoader for test set.
        
        Args:
            batch_size: Batch size
            
        Returns:
            DataLoader for test data
        """
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    
    def get_client_data_distribution(self,
                                    client_data: Dict[int, List[int]]) -> None:
        """Print data distribution across clients.
        
        Args:
            client_data: Dictionary from create_non_iid_split()
        """
        labels = np.array(self.train_dataset.targets)
        
        print("\n" + "="*60)
        print("CLIENT DATA DISTRIBUTION")
        print("="*60)
        
        for client_id in sorted(client_data.keys()):
            indices = client_data[client_id]
            client_labels = labels[indices]
            unique, counts = np.unique(client_labels, return_counts=True)
            
            dist_str = ", ".join([f"C{c}:{cnt}" for c, cnt in zip(unique, counts)])
            print(f"Client {client_id:2d}: {len(indices):4d} samples | {dist_str}")
        
        print("="*60 + "\n")
