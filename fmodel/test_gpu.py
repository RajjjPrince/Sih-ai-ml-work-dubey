"""
Quick GPU detection test script
"""

import torch
import subprocess
import sys

print("=" * 60)
print("GPU Detection Test")
print("=" * 60)

# Check PyTorch version
print(f"\n1. PyTorch Version: {torch.__version__}")
print(f"2. CUDA Available: {torch.cuda.is_available()}")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"\n✓ GPU Detected!")
    print(f"3. CUDA Version: {torch.version.cuda}")
    print(f"4. Number of GPUs: {torch.cuda.device_count()}")
    print(f"5. Current GPU: {torch.cuda.current_device()}")
    print(f"6. GPU Name: {torch.cuda.get_device_name(0)}")
    
    # GPU properties
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU Properties:")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    except Exception as e:
        print(f"  Error getting properties: {e}")
    
    # Test GPU tensor creation
    try:
        x = torch.randn(3, 3).cuda()
        print(f"\n7. ✓ GPU tensor test: SUCCESS")
        print(f"   Tensor device: {x.device}")
    except Exception as e:
        print(f"\n7. ✗ GPU tensor test: FAILED")
        print(f"   Error: {e}")
else:
    print("\n⚠ No CUDA GPU detected!")
    print("  Training will use CPU (much slower)")
    
    # Check if nvidia-smi works
    print("\nChecking NVIDIA drivers...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ nvidia-smi works - GPU hardware is detected")
            print("  But PyTorch doesn't see it - need CUDA-enabled PyTorch")
            print("\n🔧 SOLUTION:")
            print("  Run these commands:")
            print("  pip uninstall torch torchvision torchaudio")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        else:
            print("✗ nvidia-smi failed")
    except FileNotFoundError:
        print("✗ nvidia-smi not found - GPU drivers may not be installed")
        print("\n🔧 SOLUTION:")
        print("  1. Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
        print("  2. Then install CUDA-enabled PyTorch:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    except Exception as e:
        print(f"? Could not check nvidia-smi: {e}")

# Check MPS (Apple Silicon)
if hasattr(torch.backends, 'mps'):
    mps_available = torch.backends.mps.is_available()
    print(f"\n8. MPS (Apple Silicon) Available: {mps_available}")
    if mps_available:
        print("   ✓ Can use Apple GPU for acceleration")

print("\n" + "=" * 60)
if torch.cuda.is_available():
    print("✓ GPU is ready! You can train with GPU acceleration.")
else:
    print("⚠ No GPU detected. Training will use CPU (slower but works).")
print("=" * 60)


