#!/bin/bash

# Example runs for Hourglass experiments

echo "HOURGLASS EXAMPLE RUNS"
echo "====================="
echo ""
echo "These are example commands to run different Hourglass configurations."
echo "Uncomment and run them as needed."
echo ""

# Example 1: Basic run with FCFS scheduler
echo "# Example 1: Baseline with FCFS scheduler (10 clients, 5 rounds)"
echo "python main.py --num_clients 10 --num_rounds 5 --scheduler fcfs"
echo ""

# Example 2: DFF scheduler
echo "# Example 2: DFF scheduler (20 clients, 10 rounds)"
echo "python main.py --num_clients 20 --num_rounds 10 --scheduler dff"
echo ""

# Example 3: More non-IID data
echo "# Example 3: Highly non-IID data (alpha=0.01)"
echo "python main.py --num_clients 30 --num_rounds 15 --alpha 0.01"
echo ""

# Example 4: IID data (for comparison)
echo "# Example 4: IID data (alpha=10.0) - for comparison"
echo "python main.py --num_clients 10 --num_rounds 5 --alpha 10.0"
echo ""

# Example 5: Larger batch size
echo "# Example 5: Larger batch size (64)"
echo "python main.py --num_clients 15 --batch_size 64 --num_rounds 10"
echo ""

# Example 6: CPU-only mode
echo "# Example 6: CPU-only mode (no GPU)"
echo "python main.py --num_clients 5 --num_rounds 2 --use_gpu false"
echo ""

# Example 7: High-frequency evaluation
echo "# Example 7: More detailed logging"
echo "python main.py --num_clients 10 --num_rounds 20 --scheduler dff"
echo ""
