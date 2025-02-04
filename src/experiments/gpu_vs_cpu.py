import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from typing import Dict, Any
from ..models.resnet import ResNet, BasicBlock
from ..training.trainer import Trainer

def compare_gpu_cpu(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 5
) -> Dict[str, Any]:
    """Compares training performance between GPU and CPU"""
    results = {}
    
    # GPU Training
    if torch.cuda.is_available():
        print("\nTraining on GPU")
        device = torch.device("cuda")
        model_gpu = ResNet(
            img_channels=3,
            num_layers=18,
            block=BasicBlock,
            num_classes=10
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model_gpu.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        trainer = Trainer(model_gpu, criterion, optimizer, device)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        gpu_results = trainer.train(
            train_loader,
            num_epochs,
            test_loader,
            measure_time=True
        )
        
        torch.cuda.synchronize()
        gpu_total_time = time.perf_counter() - start_time
        
        gpu_results['total_time'] = gpu_total_time
        results['gpu'] = gpu_results
    else:
        print("CUDA is not available. Skipping GPU experiment.")
    
    # CPU Training
    print("\nTraining on CPU")
    device = torch.device("cpu")
    model_cpu = ResNet(
        img_channels=3,
        num_layers=18,
        block=BasicBlock,
        num_classes=10
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model_cpu.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    
    trainer = Trainer(model_cpu, criterion, optimizer, device)
    
    start_time = time.perf_counter()
    cpu_results = trainer.train(
        train_loader,
        num_epochs,
        test_loader,
        measure_time=True
    )
    cpu_total_time = time.perf_counter() - start_time
    
    cpu_results['total_time'] = cpu_total_time
    results['cpu'] = cpu_results
    
    return results 