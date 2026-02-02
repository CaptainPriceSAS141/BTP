"""Model aggregation for federated learning.

Paper Section 3.3: Federated Aggregation
Implements FedAvg aggregation (averaged across GPUs, not clients).
"""

import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np


class FedAvgAggregator:
    """Federated Averaging aggregator.
    
    Paper Section 3.3: Model Aggregation
    Aggregates server models across GPUs using FedAvg.
    NOT across clients (clients don't directly communicate in Hourglass).
    """
    
    def __init__(self, num_gpus: int = 1):
        """Initialize aggregator.
        
        Args:
            num_gpus: Number of GPUs
        """
        self.num_gpus = num_gpus
    
    def aggregate_models(self, 
                        model_states: List[Dict]) -> Dict:
        """Aggregate multiple model states using FedAvg.
        
        Formula: w_avg = (1/K) * sum(w_k) for k in K GPUs
        
        Args:
            model_states: List of state_dicts from multiple GPU models
            
        Returns:
            Aggregated state_dict
        """
        if not model_states:
            return {}
        
        if len(model_states) == 1:
            return model_states[0]
        
        # Average the parameters
        aggregated = {}
        
        for key in model_states[0].keys():
            aggregated[key] = torch.mean(
                torch.stack([state[key].float() for state in model_states]),
                dim=0
            )
        
        return aggregated
    
    def aggregate_models_weighted(self,
                                 model_states: List[Dict],
                                 weights: List[float]) -> Dict:
        """Weighted FedAvg aggregation.
        
        Formula: w_avg = sum(weight_k * w_k) for k in K GPUs
        
        Args:
            model_states: List of state_dicts
            weights: List of weights (should sum to 1.0)
            
        Returns:
            Aggregated state_dict
        """
        if not model_states:
            return {}
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        aggregated = {}
        
        for key in model_states[0].keys():
            weighted_sum = None
            
            for state, weight in zip(model_states, weights):
                if weighted_sum is None:
                    weighted_sum = weight * state[key].float()
                else:
                    weighted_sum = weighted_sum + weight * state[key].float()
            
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def client_selection(self, 
                        num_clients: int, 
                        participation_rate: float = 1.0,
                        seed: int = 42) -> List[int]:
        """Select subset of clients for current round.
        
        Paper Section 5: Client Selection
        For Hourglass, all clients typically participate.
        This is included for completeness.
        
        Args:
            num_clients: Total number of clients
            participation_rate: Fraction of clients to select (0-1)
            seed: Random seed
            
        Returns:
            List of selected client indices
        """
        np.random.seed(seed)
        
        num_selected = max(1, int(num_clients * participation_rate))
        selected = np.random.choice(num_clients, num_selected, replace=False)
        
        return selected.tolist()


def aggregate_client_models(client_models: List[nn.Module],
                           sample_counts: List[int] = None) -> Dict:
    """Aggregate client models using FedAvg.
    
    Used for evaluation/analysis purposes.
    
    Args:
        client_models: List of client models
        sample_counts: Number of samples per client (for weighted averaging)
        
    Returns:
        Aggregated state_dict
    """
    if not client_models:
        return {}
    
    # Get state dicts
    states = [model.state_dict() for model in client_models]
    
    aggregator = FedAvgAggregator(len(client_models))
    
    if sample_counts is None:
        # Uniform averaging
        return aggregator.aggregate_models(states)
    else:
        # Weighted by sample count
        return aggregator.aggregate_models_weighted(states, sample_counts)
