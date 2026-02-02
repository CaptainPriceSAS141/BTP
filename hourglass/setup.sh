#!/bin/bash

# Hourglass Setup Script
# This script sets up the environment and installs all dependencies

set -e

echo "=========================================="
echo "Hourglass Setup Script"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Installing Miniconda..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
    else
        echo "Unsupported OS. Please install Python3 manually."
        exit 1
    fi
    
    INSTALLER="miniconda_installer.sh"
    wget "$CONDA_URL" -O "$INSTALLER"
    bash "$INSTALLER" -b -p "$HOME/miniconda3"
    rm "$INSTALLER"
    
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

echo "Python version: $(python3 --version)"

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Virtual environment activated."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Detect GPU
echo "Detecting GPU..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No GPU detected. CPU-only mode.")
EOF

echo "=========================================="
echo "Setup complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "=========================================="
