# Configuration constants for Hourglass

# Model Architecture
SPLIT_POINT = 5  # Split ResNet-18 after layer 5 (early layers on client)
MODEL_NAME = "resnet18"
NUM_CLASSES = 10  # CIFAR-10

# Training Parameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_NUM_ROUNDS = 5
DEFAULT_NUM_CLIENTS = 10

# Data Distribution
DEFAULT_ALPHA = 0.1  # Dirichlet parameter for non-IID (lower = more non-IID)
TRAIN_TEST_SPLIT = 0.8

# Scheduling
SCHEDULER_FCFS = "fcfs"
SCHEDULER_DFF = "dff"
DEFAULT_SCHEDULER = SCHEDULER_FCFS

# Feature Similarity (for DFF)
FEATURE_EMBEDDING_DIM = 128
SIMILARITY_THRESHOLD = 0.5

# LSH Configuration
LSH_NUM_BUCKETS = 10
LSH_HASH_FUNCS = 5

# Device
DEFAULT_DEVICE = "cuda"  # Will fall back to "cpu" if not available

# Logging
DEFAULT_LOG_DIR = "logs/"
LOG_INTERVAL = 1  # Log every N rounds
