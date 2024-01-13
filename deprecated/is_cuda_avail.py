import torch

def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

# Run the check
check_cuda_availability()
