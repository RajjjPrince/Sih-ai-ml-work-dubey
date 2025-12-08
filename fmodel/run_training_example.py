"""
Example script to run TCN-GNN model training

This is a simple example showing how to train the model.
"""

from pathlib import Path
import torch
from train_tcn_gnn import train
from check_gpu import check_gpu_status

# Configuration
DATA_DIR = Path("Data_SIH_2025_with_blh/Data_SIH_2025_with_blh/with_satellite")
SITE_IDS = [1, 2, 3, 4, 5, 6, 7]
SAVE_DIR = Path("models/tcn_gnn")

# Training parameters
SEQ_LENGTH = 24  # 24-hour sequence window
PREDICTION_HORIZON = 1  # Predict 1 hour ahead
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
LOSS_TYPE = 'mae'  # or 'huber'
PATIENCE = 10  # Early stopping patience

if __name__ == '__main__':
    print("=" * 60)
    print("TCN-GNN Model Training")
    print("=" * 60)
    
    # Check GPU status first
    print("\nChecking GPU status...")
    gpu_available = check_gpu_status()
    print()
    
    # Adjust batch size if GPU is available
    if gpu_available and BATCH_SIZE < 64:
        print(f"⚠ GPU detected but batch_size is {BATCH_SIZE}")
        print(f"  Consider increasing batch_size to 64-128 for faster training")
        print(f"  (Current batch_size will be used)\n")
    
    print(f"Data directory: {DATA_DIR}")
    print(f"Sites: {SITE_IDS}")
    print(f"Sequence length: {SEQ_LENGTH} hours")
    print(f"Prediction horizon: {PREDICTION_HORIZON} hours")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {NUM_EPOCHS}")
    print(f"Loss type: {LOSS_TYPE}")
    print(f"Early stopping patience: {PATIENCE}")
    
    # Use mixed precision if GPU is available
    USE_MIXED_PRECISION = gpu_available
    if USE_MIXED_PRECISION:
        print(f"Mixed precision: Enabled (faster GPU training)")
    
    print("=" * 60)
    
    # Train model
    model, history = train(
        data_dir=DATA_DIR,
        site_ids=SITE_IDS,
        seq_length=SEQ_LENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        loss_type=LOSS_TYPE,
        patience=PATIENCE,
        save_dir=SAVE_DIR,
        use_mixed_precision=USE_MIXED_PRECISION
    )
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {SAVE_DIR}")
