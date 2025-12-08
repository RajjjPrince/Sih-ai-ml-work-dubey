# GPU Optimization Guide

## Quick Check

Run this command to check if your GPU is detected:

```bash
python check_gpu.py
```

This will show:
- Whether CUDA is available
- GPU name and memory
- CUDA version
- Memory usage

## GPU Detection

The training script automatically detects and uses GPU if available. You'll see output like:

```
✓ CUDA available! Using GPU: NVIDIA GeForce RTX 3090
  GPU Memory: 24.00 GB
Using device: cuda
✓ Model moved to GPU: cuda:0
```

If no GPU is detected, you'll see:

```
⚠ No GPU detected. Using CPU (training will be slow)
  Consider using a GPU-enabled environment for faster training
Using device: cpu
⚠ Model on CPU (training will be slow)
```

## GPU Optimizations Enabled

When GPU is detected, the following optimizations are automatically enabled:

1. **CUDA Device**: Model and data moved to GPU
2. **Non-blocking Transfers**: Faster data loading to GPU
3. **Pin Memory**: Faster CPU-to-GPU data transfer
4. **Multiple Workers**: Parallel data loading (4 workers)
5. **cuDNN Benchmarking**: Faster convolutions (if input sizes are constant)
6. **Mixed Precision Training**: Optional (use `--use_mixed_precision` flag)

## Using Mixed Precision Training

Mixed precision training can speed up GPU training by 1.5-2x:

```bash
python train_tcn_gnn.py --use_mixed_precision
```

This uses FP16 for forward pass and FP32 for backward pass, reducing memory usage and increasing training speed.

## Recommended Batch Sizes

- **CPU**: 8-16 (smaller batches)
- **GPU (8GB)**: 32-64
- **GPU (16GB+)**: 64-128

You can adjust batch size:

```bash
python train_tcn_gnn.py --batch_size 64
```

## Troubleshooting

### GPU Not Detected

1. **Check CUDA Installation**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Install CUDA-enabled PyTorch**:
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify NVIDIA Driver**:
   ```bash
   nvidia-smi
   ```

### Out of Memory Errors

If you get CUDA out of memory errors:

1. **Reduce batch size**:
   ```bash
   python train_tcn_gnn.py --batch_size 16
   ```

2. **Use mixed precision**:
   ```bash
   python train_tcn_gnn.py --use_mixed_precision --batch_size 32
   ```

3. **Reduce sequence length**:
   ```bash
   python train_tcn_gnn.py --seq_length 12
   ```

### Slow Training on GPU

If training is still slow even with GPU:

1. **Check GPU utilization**:
   ```bash
   # In another terminal, run:
   watch -n 1 nvidia-smi
   ```
   GPU utilization should be >80% during training

2. **Increase batch size** (if memory allows):
   ```bash
   python train_tcn_gnn.py --batch_size 128
   ```

3. **Enable mixed precision**:
   ```bash
   python train_tcn_gnn.py --use_mixed_precision
   ```

4. **Check data loading**: Make sure `num_workers > 0` in data loader (automatically set when GPU is detected)

## Performance Comparison

Expected training speeds (approximate):

| Device | Batch Size | Time per Epoch | Notes |
|--------|------------|----------------|-------|
| CPU (Intel i7) | 16 | ~30-60 min | Very slow |
| GPU (RTX 3060) | 32 | ~2-5 min | Good |
| GPU (RTX 3090) | 64 | ~1-2 min | Fast |
| GPU (RTX 4090) | 128 | ~30-60 sec | Very fast |

*Times vary based on dataset size and model complexity*

## Apple Silicon (M1/M2/M3) Support

The code also supports Apple Silicon GPUs via MPS:

```
✓ MPS (Apple Silicon) available! Using Apple GPU
Using device: mps
```

MPS support is automatically detected and used if available.

## Monitoring GPU Usage

During training, you can monitor GPU usage:

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Or use Python
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')"
```

## Summary

The training script automatically:
- ✅ Detects GPU availability
- ✅ Moves model and data to GPU
- ✅ Optimizes data loading for GPU
- ✅ Uses mixed precision if requested
- ✅ Provides clear status messages

Just run the training script - GPU optimizations are automatic!
