# QUICKSTART Guide - Hourglass

## One-Minute Setup

```bash
cd hourglass
bash setup.sh
source venv/bin/activate
python main.py --num_clients 10 --num_rounds 5
```

## What Just Happened?

You ran Hourglass split federated learning:
- **10 clients** with non-IID data downloaded from CIFAR-10
- **5 federated rounds** of training
- **FCFS scheduler** processed client requests
- Results saved to `logs/metrics.json`

## Verify Installation

```bash
# Check all modules work
python3 << 'EOF'
from models.client_model import ClientModelPartition
from models.server_model import ServerModelPartition
from server.trainer import ServerTrainer
from server.scheduler import create_scheduler
from datasets.cifar10 import CIFAR10NonIID
import torch
print("✓ All modules imported successfully")
print(f"✓ GPU Available: {torch.cuda.is_available()}")
EOF
```

## Common Commands

### Baseline (FCFS)
```bash
python main.py --num_clients 10 --num_rounds 5 --scheduler fcfs
```

### Advanced (DFF Scheduler)
```bash
python main.py --num_clients 20 --num_rounds 10 --scheduler dff
```

### Highly Non-IID Data
```bash
python main.py --num_clients 15 --alpha 0.01 --num_rounds 10
```

### IID Data (For Comparison)
```bash
python main.py --num_clients 10 --alpha 10.0 --num_rounds 5
```

### CPU-Only Mode
```bash
python main.py --num_clients 5 --use_gpu false --num_rounds 3
```

## What Each Argument Does

- `--num_clients`: Number of federated clients (default: 10)
- `--num_rounds`: Federated learning rounds (default: 5)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: SGD learning rate (default: 0.01)
- `--scheduler`: FCFS or DFF (default: fcfs)
- `--alpha`: Dirichlet for non-IID (lower = more non-IID, default: 0.1)
- `--log_dir`: Where to save metrics (default: logs/)
- `--seed`: Random seed for reproducibility (default: 42)

## Understanding the Output

```
Round 1: Loss=1.2218, Acc=0.1103, Time=14.39s
```
- **Loss**: Average training loss for the round
- **Acc**: Test set accuracy after the round
- **Time**: Seconds to complete the round

## Folder Structure

```
hourglass/
├── main.py                    # Entry point
├── datasets/cifar10.py       # Data loading + non-IID partitioning
├── models/                   # Client/Server models
├── clients/client.py         # Client-side logic
├── server/                   # Scheduler, trainer, aggregator, LSH
└── utils/                    # Metrics, logging, config
```

## Next Steps

1. **Run a full experiment**: `python main.py --num_clients 50 --num_rounds 20 --scheduler dff`
2. **Compare schedulers**: Run with `--scheduler fcfs` and `--scheduler dff`, compare metrics
3. **Analyze results**: Load `logs/metrics.json` and plot accuracy vs rounds
4. **Extend the code**: Add differential privacy, compression, multi-GPU support

## Troubleshooting

**GPU not detected?**
```bash
# Check CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Force CPU
python main.py --use_gpu false
```

**Out of memory?**
```bash
# Reduce batch size and clients
python main.py --batch_size 16 --num_clients 5
```

**Data download slow?**
Data is cached in `data/cifar-10-batches-py/` after first download.

## Paper Mapping

| Paper Section | Implementation |
|---|---|
| 3.1: Split FL Workflow | `main.py::train_round()`, `clients/client.py`, `server/trainer.py` |
| 3.2: Hourglass Design | `server/trainer.py` (single shared model) |
| 4.1: FCFS & DFF | `server/scheduler.py` |
| 4.2: LSH Clustering | `server/lsh.py` |
| 5: Experiments | `datasets/cifar10.py` (non-IID), `utils/metrics.py` |

## Contact & Support

For questions about the implementation, refer to inline code comments that map to paper sections.
