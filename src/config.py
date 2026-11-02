"""Paths and hyperparameters for the bone age project."""
import os
from pathlib import Path

# Detect Colab
IN_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")

# Project root (folder containing src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = ROOT_DIR / "data"
TRAIN_CSV = DATA_DIR / "boneage-training-dataset.csv"
TRAIN_IMG_DIR = DATA_DIR / "boneage-training-dataset" / "boneage-training-dataset"
CACHE_SIZE = 640  # images are pre-resized once (longest side) to speed up loading
CACHE_DIR = DATA_DIR / f"resized_{CACHE_SIZE}"
SPLITS_CSV = DATA_DIR / "splits.csv"

# Outputs (on Colab, save to Drive so checkpoints survive disconnects)
if IN_COLAB and os.path.exists("/content/drive/MyDrive"):
    OUTPUT_DIR = Path("/content/drive/MyDrive/bone-age/outputs")
else:
    OUTPUT_DIR = ROOT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
SEED = 42

# Split
VAL_FRAC = 0.10
TEST_FRAC = 0.10
N_AGE_BINS = 10

# Data loading
IMG_SIZE = 384
BATCH_SIZE = 16
NUM_WORKERS = 4

# Model
BACKBONE = "efficientnet_b0"
SEX_DIM = 16
HIDDEN_DIM = 256
DROPOUT = 0.2

# Training
EPOCHS = 30
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
