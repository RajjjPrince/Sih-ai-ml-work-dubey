"""
Utility script to check GPU/CUDA availability and status
"""

import torch

def check_gpu_status():
    """Check and print GPU/CUDA status"""
    print("=" * 60)
    print("GPU/CUDA Status Check")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"\n✓ GPU Detected!")
        print(f"  Device Count: {torch.cuda.device_count()}")
        print(f"  Current Device: {torch.cuda.current_device()}")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        
        # GPU properties
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU Properties:")
        print(f"  Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        
        # Memory info
        print(f"\nMemory Info:")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        # CUDA version
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
    else:
        print("\n⚠ No CUDA GPU detected!")
        print("  Training will use CPU (much slower)")
        print("\nTo use GPU:")
        print("  1. Install CUDA-enabled PyTorch:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  2. Ensure you have NVIDIA GPU with CUDA support")
        print("  3. Install CUDA toolkit from NVIDIA")
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        mps_available = torch.backends.mps.is_available()
        print(f"\nMPS (Apple Silicon) Available: {mps_available}")
        if mps_available:
            print("  ✓ Can use Apple GPU for acceleration")
    
    print("=" * 60)
    
    return cuda_available

if __name__ == '__main__':
    check_gpu_status()
