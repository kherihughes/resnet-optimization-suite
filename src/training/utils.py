import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def get_default_transform():
    """Returns the default transform for CIFAR10 training"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), 
            (0.2023, 0.1994, 0.2010)
        )
    ])

def measure_data_loading_time(train_loader: DataLoader) -> float:
    """Measures the time taken to iterate through the entire dataloader"""
    start_time = time.perf_counter()
    for _ in train_loader:
        pass
    return time.perf_counter() - start_time

def get_device(use_cuda: bool = True) -> torch.device:
    """Returns the appropriate device based on CUDA availability"""
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def count_parameters(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 