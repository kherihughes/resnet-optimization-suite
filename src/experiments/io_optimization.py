import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import List, Tuple
from ..training.utils import get_default_transform, measure_data_loading_time

def find_optimal_workers(
    dataset_path: str = './data',
    batch_size: int = 128,
    worker_settings: List[int] = None
) -> Tuple[int, List[Tuple[int, float]]]:
    """
    Finds the optimal number of workers for data loading
    Returns the optimal worker count and results for all tested settings
    """
    if worker_settings is None:
        worker_settings = [0, 4, 8, 12, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    
    train_dataset = datasets.CIFAR10(
        root=dataset_path,
        train=True,
        download=True,
        transform=get_default_transform()
    )
    
    results = []
    best_time = float('inf')
    optimal_workers = 0
    
    for workers in worker_settings:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers
        )
        
        data_loading_time = measure_data_loading_time(train_loader)
        results.append((workers, data_loading_time))
        
        print(f"num_workers={workers} - Total Data Loading Time: {data_loading_time:.2f}s")
        
        if data_loading_time < best_time:
            best_time = data_loading_time
            optimal_workers = workers
    
    return optimal_workers, results 