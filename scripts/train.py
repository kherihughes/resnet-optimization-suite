import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import argparse
from typing import Dict, Any

from src.models.resnet import ResNet, BasicBlock
from src.training.utils import get_default_transform, get_device
from src.training.trainer import Trainer
from src.experiments import (
    io_optimization,
    optimizer_comparison,
    gpu_vs_cpu,
    compile_acceleration
)

def get_dataloaders(
    data_path: str,
    batch_size: int = 128,
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """Creates train and test dataloaders"""
    transform = get_default_transform()
    
    train_dataset = datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return {'train': train_loader, 'test': test_loader}

def main():
    parser = argparse.ArgumentParser(description='ResNet Training and Optimization')
    parser.add_argument('--experiment', type=str, required=True,
                      choices=['io', 'gpu', 'optimizer', 'compile'],
                      help='Experiment to run')
    parser.add_argument('--use_cuda', type=bool, default=True,
                      help='Use CUDA if available')
    parser.add_argument('--data_path', type=str, default='./data',
                      help='Path to the CIFAR10 data')
    parser.add_argument('--num_workers', type=int, default=2,
                      help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Training batch size')
    parser.add_argument('--profile', action='store_true',
                      help='Enable profiling')
    args = parser.parse_args()
    
    device = get_device(args.use_cuda)
    
    if args.experiment == 'io':
        optimal_workers, results = io_optimization.find_optimal_workers(
            args.data_path,
            args.batch_size
        )
        print(f"\nOptimal number of workers: {optimal_workers}")
        
    elif args.experiment == 'optimizer':
        dataloaders = get_dataloaders(
            args.data_path,
            args.batch_size,
            args.num_workers
        )
        results = optimizer_comparison.compare_optimizers(
            dataloaders['train'],
            dataloaders['test'],
            device
        )
        
    elif args.experiment == 'gpu':
        dataloaders = get_dataloaders(
            args.data_path,
            args.batch_size,
            args.num_workers
        )
        results = gpu_vs_cpu.compare_gpu_cpu(
            dataloaders['train'],
            dataloaders['test']
        )
        
    elif args.experiment == 'compile':
        if not torch.cuda.is_available():
            print("CUDA is required for compile experiment")
            return
            
        dataloaders = get_dataloaders(
            args.data_path,
            args.batch_size,
            args.num_workers
        )
        results = compile_acceleration.compare_compile_modes(
            dataloaders['train'],
            dataloaders['test'],
            device
        )

if __name__ == '__main__':
    main() 