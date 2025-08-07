import torch
print(torch.cuda.is_available())  # Should return True if CUDA is working
print(torch.cuda.device_count())  # Number of available GPUs
print(torch.cuda.get_device_name(0))  # Name of the first GPU
print(torch.version.cuda)  # Installed CUDA version compatible with PyTorch