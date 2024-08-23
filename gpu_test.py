import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    # Get the number of GPU devices
    print("Number of GPU devices:", torch.cuda.device_count())
    # Get the current device
    print("Current device:", torch.cuda.current_device())
    # Get the name of the current device
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available.")
