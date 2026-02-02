# Hourglass: Enabling Efficient Split Federated Learning with Data Parallelism

## Overview

This is a research reproduction of the Hourglass system from the paper **"Hourglass: Enabling Efficient Split Federated Learning with Data Parallelism"** (EuroSys 2025).

### Key Innovation
Hourglass enables efficient **split federated learning** by maintaining a **single shared server-side model** across multiple GPUs, rather than creating one model per client as in traditional SplitFed. This reduces memory overhead and enables intelligent scheduling of client requests.

## Architecture

### Components

1. **Client-side Model Partition**: Early layers of ResNet-18 run on clients
2. **Server-side Model Partition**: Deeper layers run on server GPUs
3. **Split Federated Learning Workflow**: Clients send intermediate features to server
4. **Hourglass Shared Model Logic**: One server model per GPU, not per client
5. **Schedulers**:
   - FCFS: First Come First Serve
   - DFF: Dissimilar Feature First
6. **LSH-based Feature Clustering**: Advanced scheduling for heterogeneous features

### System Design

```
Client 1 -> [Early layers] -> Intermediate Features
Client 2 -> [Early layers] -> Intermediate Features  --|
...                                                   |---> Scheduler (FCFS/DFF) ---> GPU1: [Server Model]
Client N -> [Early layers] -> Intermediate Features  |    |
                                                     |    |---> GPU2: [Server Model]
                                                     |
                                                     |---> GPU3: [Server Model]
```

## Project Structure

```
hourglass/
├── README.md                 # This file
├── setup.sh                  # Environment setup script
├── requirements.txt          # Python dependencies
├── main.py                   # Main training loop
│
├── datasets/
│   └── cifar10.py           # CIFAR-10 data loading with non-IID partitioning
│
├── models/
│   ├── client_model.py      # Client-side model (early layers)
│   ├── server_model.py      # Server-side model (deep layers)
│   └── split_model.py       # Complete model split utilities
│
├── clients/
│   └── client.py            # Client logic (forward pass, feature sending)
│
├── server/
│   ├── scheduler.py         # FCFS and DFF schedulers
│   ├── trainer.py           # Server-side training (backward pass)
│   ├── aggregator.py        # FedAvg aggregation
│   └── lsh.py              # LSH-based feature clustering
│
└── utils/
    ├── metrics.py           # Accuracy, loss, convergence metrics
    ├── logger.py            # Logging and result tracking
    └── config.py            # Configuration constants
```

## Installation & Setup

### Quick Start

```bash
# Navigate to hourglass directory
cd hourglass

# Run setup script (installs Python, creates venv, installs dependencies)
bash setup.sh

# Activate environment
source venv/bin/activate
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Experiment

### Default Configuration (Single GPU, CIFAR-10)

```bash
# Run with default settings
python main.py

# Run with FCFS scheduler
python main.py --scheduler fcfs

# Run with DFF scheduler
python main.py --scheduler dff

# Run with custom settings
python main.py --num_clients 20 --num_rounds 10 --batch_size 32 --scheduler dff
```

### Available Arguments

```
--num_clients        Number of federated clients (default: 10)
--num_rounds         Number of federated learning rounds (default: 5)
--local_epochs       Local training epochs per client (default: 1)
--batch_size         Batch size for training (default: 32)
--learning_rate      Learning rate (default: 0.01)
--scheduler          Scheduler type: fcfs, dff (default: fcfs)
--use_gpu            Use GPU if available (default: True)
--seed               Random seed (default: 42)
--log_dir            Directory to save logs (default: logs/)
--non_iid            Non-IID data distribution (default: True)
--alpha              Dirichlet parameter for non-IID (default: 0.1, lower=more non-IID)
```

### Example Runs

```bash
# Baseline comparison (10 clients, FCFS)
python main.py --num_clients 10 --scheduler fcfs --num_rounds 10

# Advanced setup (DFF with LSH clustering)
python main.py --num_clients 50 --scheduler dff --num_rounds 20

# CPU-only mode
python main.py --use_gpu false --num_clients 5
```

## Key Implementation Details

### Split Federated Learning Workflow

1. **Client Forward Pass**: 
   - Client runs early layers of ResNet-18 on local data
   - Produces intermediate features (activation maps)

2. **Feature Transmission**:
   - Client sends intermediate features to server scheduler
   - Server scheduler decides which GPU processes the request

3. **Server Training**:
   - Server-side model (deep layers) receives features
   - Runs forward pass through remaining layers
   - Computes loss
   - Runs backward pass
   - Returns gradients to client

4. **Client Backward Pass**:
   - Client receives gradients from server
   - Completes backward pass through early layers
   - Updates client-side model

5. **Model Aggregation**:
   - FedAvg aggregation (only across GPUs, not clients)
   - Each GPU maintains one shared model copy

### Hourglass vs SplitFed

| Aspect | SplitFed | Hourglass |
|--------|----------|-----------|
| Server Models | One per client | One per GPU (shared) |
| Memory Overhead | O(num_clients * model_size) | O(num_gpus * model_size) |
| Scheduling | FCFS only | FCFS, DFF, LSH-based |
| Scalability | Limited to ~10 clients | Tested with 50+ clients |

### Scheduling Algorithms

**FCFS (First Come First Serve)**:
- Processes client requests in arrival order
- Simple, predictable
- May be inefficient for heterogeneous features

**DFF (Dissimilar Feature First)**:
- Measures feature similarity using embeddings
- Prioritizes dissimilar feature batches
- Reduces server model cache misses
- Improves GPU utilization

**LSH (Locality Sensitive Hashing)**:
- Clusters features based on Euclidean distance
- Async-friendly (no need to wait for all clients)
- Assigns features to GPUs probabilistically
- Advanced scheduling strategy

## Metrics & Evaluation

The system logs:
- **Per-round loss**: Training loss at each FL round
- **Per-round accuracy**: Top-1 classification accuracy
- **Time-to-accuracy**: Time to reach target accuracy
- **Memory usage**: GPU/CPU memory consumption
- **Throughput**: Samples processed per second
- **Communication cost**: Number of features transmitted

Results are saved to `logs/` directory as JSON and plotted to PNG.

## Code Mapping to Paper

- **Section 3.1**: Split federated learning workflow → [clients/client.py](clients/client.py) + [server/trainer.py](server/trainer.py)
- **Section 3.2**: Hourglass shared model design → [server/trainer.py](server/trainer.py) (single model per GPU)
- **Section 4.1**: FCFS & DFF scheduling → [server/scheduler.py](server/scheduler.py)
- **Section 4.2**: LSH-based clustering → [server/lsh.py](server/lsh.py)
- **Section 5**: Experiments setup → [main.py](main.py)

## GPU Support

### Single GPU (Tested)
- Works out-of-the-box
- All client requests served by GPU 0

### Multi-GPU (Extension)
- [TODO] Implement distributed server models across GPUs
- [TODO] Add gradient synchronization between GPU replicas
- [TODO] Benchmark scaling efficiency

## Known Limitations & TODOs

- [ ] Multi-GPU distributed training
- [ ] Gradient compression for communication efficiency
- [ ] Asynchronous client-server communication
- [ ] Privacy-preserving gradients (differential privacy)
- [ ] Support for larger models (ResNet-50, ViT)
- [ ] Benchmark on ImageNet

## References

> Hourglass: Enabling Efficient Split Federated Learning with Data Parallelism  
> EuroSys 2025  
> [Link to paper will be added]

## Contact & Questions

For implementation questions, refer to inline code comments mapping to paper sections.

## License

Research reproduction. Follow original paper's licensing terms.
