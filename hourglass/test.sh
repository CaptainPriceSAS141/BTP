#!/bin/bash

# Test script for Hourglass implementation

set -e

echo "=========================================="
echo "HOURGLASS TEST SUITE"
echo "=========================================="

# Check if we're in the hourglass directory
if [ ! -f "main.py" ]; then
    echo "Error: Please run this script from the hourglass/ directory"
    exit 1
fi

echo "✓ Found main.py"

# Test 1: Check Python imports
echo ""
echo "Test 1: Checking Python imports..."
python3 << 'EOF'
import sys
try:
    import torch
    print(f"  ✓ torch {torch.__version__}")
    
    import torchvision
    print(f"  ✓ torchvision {torchvision.__version__}")
    
    import numpy as np
    print(f"  ✓ numpy {np.__version__}")
    
    import sklearn
    print(f"  ✓ scikit-learn {sklearn.__version__}")
    
    print("  All imports successful!")
except ImportError as e:
    print(f"  ✗ Import error: {e}")
    sys.exit(1)
EOF

# Test 2: Check GPU availability
echo ""
echo "Test 2: Checking GPU availability..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  ℹ No GPU available (CPU-only mode)")
EOF

# Test 3: Test dataset loading
echo ""
echo "Test 3: Testing dataset loading..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from datasets.cifar10 import CIFAR10NonIID

print("  Loading CIFAR-10...")
dataset = CIFAR10NonIID()
print(f"  ✓ Train dataset: {len(dataset.train_dataset)} samples")
print(f"  ✓ Test dataset: {len(dataset.test_dataset)} samples")

print("  Creating non-IID split (10 clients)...")
client_data = dataset.create_non_iid_split(num_clients=10, alpha=0.1)
print(f"  ✓ Created split for {len(client_data)} clients")
EOF

# Test 4: Test model creation
echo ""
echo "Test 4: Testing model creation..."
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
import torch
from models.client_model import ClientModelPartition
from models.server_model import ServerModelPartition

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"  Creating models on {device}...")
client_model = ClientModelPartition()
server_model = ServerModelPartition()

client_model = client_model.to(device)
server_model = server_model.to(device)

print(f"  ✓ Client model created")
print(f"  ✓ Server model created")

# Test forward pass
batch_size = 4
x = torch.randn(batch_size, 3, 32, 32).to(device)
print(f"  Testing forward pass with batch size {batch_size}...")
features = client_model(x)
print(f"  ✓ Client features shape: {features.shape}")
logits = server_model(features)
print(f"  ✓ Server output shape: {logits.shape}")
EOF

# Test 5: Quick training run (2 clients, 1 round)
echo ""
echo "Test 5: Quick training run (2 clients, 1 round)..."
python3 main.py \
    --num_clients 2 \
    --num_rounds 1 \
    --batch_size 16 \
    --learning_rate 0.01 \
    --scheduler fcfs \
    --log_dir test_logs/ \
    --seed 42

echo ""
echo "=========================================="
echo "ALL TESTS PASSED!"
echo "=========================================="
