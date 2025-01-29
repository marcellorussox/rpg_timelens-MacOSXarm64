import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")