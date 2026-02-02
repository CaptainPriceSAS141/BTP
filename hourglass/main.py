"""Main training loop for Hourglass split federated learning."""

import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import time
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.cifar10 import CIFAR10NonIID
from models.client_model import ClientModelPartition
from models.server_model import ServerModelPartition
from clients.client import FederatedClient
from server.trainer import ServerTrainer
from server.scheduler import create_scheduler
from server.aggregator import FedAvgAggregator
from server.lsh import LSHClustering
from utils.metrics import MetricsLogger
from utils.logger import setup_logger
from utils.config import *


def setup_device() -> Tuple[torch.device, bool]:
    """Setup device (GPU or CPU).
    
    Returns:
        (device, has_gpu)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        return device, True
    else:
        device = torch.device("cpu")
        print("✓ Using CPU (no GPU detected)")
        return device, False


def create_clients(num_clients: int,
                  device: torch.device,
                  learning_rate: float) -> List[FederatedClient]:
    """Create federated clients.
    
    Args:
        num_clients: Number of clients
        device: Device (GPU/CPU)
        learning_rate: Learning rate
        
    Returns:
        List of FederatedClient instances
    """
    clients = []
    
    for client_id in range(num_clients):
        # Create client-side model
        client_model = ClientModelPartition(split_layer=SPLIT_POINT)
        
        # Create client
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            device=device,
            learning_rate=learning_rate
        )
        
        clients.append(client)
    
    return clients


def evaluate(clients: List[FederatedClient],
            server_trainer: ServerTrainer,
            test_loader: torch.utils.data.DataLoader,
            device: torch.device) -> Tuple[float, float]:
    """Evaluate federated model on test set.
    
    Args:
        clients: List of client models
        server_trainer: Server trainer
        test_loader: Test data loader
        device: Device
        
    Returns:
        (accuracy, loss)
    """
    # Use first client's model as representative
    client = clients[0]
    server = server_trainer
    
    client.model.eval()
    server.eval_mode()
    
    total_correct = 0
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Client forward pass
            intermediate_features = client.model(batch_x)
            
            # Server forward pass
            logits = server.model(intermediate_features)
            
            # Compute loss
            loss = nn.CrossEntropyLoss()(logits, batch_y)
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == batch_y).sum().item()
            
            total_correct += correct
            total_loss += loss.item() * batch_y.size(0)
            total_samples += batch_y.size(0)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return accuracy, avg_loss


def train_round(clients: List[FederatedClient],
               server_trainer: ServerTrainer,
               scheduler,
               dataset: CIFAR10NonIID,
               client_data: Dict,
               batch_size: int,
               device: torch.device) -> float:
    """Execute one federated learning round.
    
    Paper Section 3.1: Split Federated Learning Workflow
    
    Args:
        clients: List of federated clients
        server_trainer: Server trainer
        scheduler: Request scheduler (FCFS/DFF)
        dataset: CIFAR-10 dataset
        client_data: Client data partition
        batch_size: Batch size
        device: Device
        
    Returns:
        Average loss for the round
    """
    server_trainer.reset_metrics()
    scheduler.reset()
    
    round_loss = 0.0
    round_samples = 0
    
    # Process each client
    for client_id, client in enumerate(clients):
        # Get client's data
        train_loader = dataset.get_client_train_loader(
            client_id, client_data, batch_size
        )
        
        # Client local training with server interaction
        client.model.train()
        server_trainer.train_mode()
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Step 1: Client forward pass
            client.model.eval()
            with torch.no_grad():
                intermediate_features = client.model(batch_x)
            
            # Step 2: Add to scheduler (for advanced scheduling demo)
            scheduler.add_request(client_id, intermediate_features)
            
            # Step 3: Server forward-backward pass
            gradients, loss = server_trainer.forward_backward_pass(
                intermediate_features, batch_y
            )
            
            # Step 4: Client backward pass (if gradients exist)
            if gradients is not None and gradients.abs().sum() > 0:
                # Recompute intermediate features with gradient tracking
                client.model.train()
                intermediate_features_grad = client.model(batch_x)
                
                # Backward on client
                intermediate_features_grad.backward(gradients)
                client.optimizer.step()
                client.optimizer.zero_grad()
            
            round_loss += loss * batch_y.size(0)
            round_samples += batch_y.size(0)
    
    return round_loss / round_samples if round_samples > 0 else 0.0


def main(args):
    """Main training loop.
    
    Args:
        args: Command-line arguments
    """
    print("="*70)
    print("HOURGLASS: Split Federated Learning with Data Parallelism")
    print("="*70)
    
    # Setup
    device, has_gpu = setup_device()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if has_gpu:
        torch.cuda.manual_seed(args.seed)
    
    # Logger
    logger = setup_logger(args.log_dir)
    metrics = MetricsLogger(args.log_dir)
    
    print(f"\n[CONFIG]")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Scheduler: {args.scheduler.upper()}")
    print(f"  Non-IID (alpha): {args.alpha}")
    print()
    
    # Load dataset
    print("[LOADING DATASET]")
    dataset = CIFAR10NonIID(seed=args.seed)
    
    # Create non-IID split
    client_data = dataset.create_non_iid_split(
        num_clients=args.num_clients,
        alpha=args.alpha
    )
    dataset.get_client_data_distribution(client_data)
    
    # Create clients
    print("[CREATING CLIENTS]")
    clients = create_clients(
        num_clients=args.num_clients,
        device=device,
        learning_rate=args.learning_rate
    )
    print(f"✓ Created {len(clients)} clients")
    print(f"  Client model parameters: {clients[0].count_parameters():,}")
    
    # Create server
    print("\n[CREATING SERVER]")
    server_model = ServerModelPartition(split_layer=SPLIT_POINT)
    server_trainer = ServerTrainer(
        model=server_model,
        device=device,
        learning_rate=args.learning_rate
    )
    print(f"✓ Server model created")
    print(f"  Server model parameters: {server_trainer.count_parameters():,}")
    
    # Create scheduler
    print(f"\n[SCHEDULER]")
    scheduler = create_scheduler(args.scheduler, num_gpus=1)
    print(f"✓ {args.scheduler.upper()} scheduler created")
    
    # Get test loader
    test_loader = dataset.get_test_loader(batch_size=args.batch_size)
    
    # Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    start_time = time.time()
    
    for round_num in range(args.num_rounds):
        round_start = time.time()
        
        # Train one round
        loss = train_round(
            clients=clients,
            server_trainer=server_trainer,
            scheduler=scheduler,
            dataset=dataset,
            client_data=client_data,
            batch_size=args.batch_size,
            device=device
        )
        
        # Evaluate
        accuracy, test_loss = evaluate(
            clients=clients,
            server_trainer=server_trainer,
            test_loader=test_loader,
            device=device
        )
        
        round_time = time.time() - round_start
        
        # Log metrics
        metrics.log_round(
            round_num=round_num + 1,
            loss=loss,
            accuracy=accuracy,
            total_time=round_time
        )
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    metrics.print_summary()
    
    # Save metrics
    metrics.save()
    
    print(f"Total training time: {total_time:.2f}s")
    print(f"Logs saved to: {args.log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hourglass Split Federated Learning"
    )
    
    # Training parameters
    parser.add_argument("--num_clients", type=int, default=DEFAULT_NUM_CLIENTS,
                       help="Number of federated clients")
    parser.add_argument("--num_rounds", type=int, default=DEFAULT_NUM_ROUNDS,
                       help="Number of federated learning rounds")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                       help="Learning rate for SGD")
    parser.add_argument("--local_epochs", type=int, default=DEFAULT_LOCAL_EPOCHS,
                       help="Local epochs per client")
    
    # Scheduling
    parser.add_argument("--scheduler", type=str, default=DEFAULT_SCHEDULER,
                       choices=["fcfs", "dff"],
                       help="Scheduling strategy")
    
    # Data distribution
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                       help="Dirichlet parameter for non-IID (lower=more non-IID)")
    
    # Device
    parser.add_argument("--use_gpu", type=bool, default=True,
                       help="Use GPU if available")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR,
                       help="Directory to save logs")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    main(args)
