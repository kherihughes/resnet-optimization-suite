import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List
from ..models.resnet import ResNet, BasicBlock
from ..training.trainer import Trainer

def compare_compile_modes(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10
) -> Dict[str, Any]:
    """Compares different torch.compile modes"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for compile experiments")
        
    compile_modes = ['eager', 'default', 'reduce-overhead', 'max-autotune']
    results = {}
    
    for mode in compile_modes:
        print(f"\nRunning with compile mode: {mode}")
        
        model = ResNet(
            img_channels=3,
            num_layers=18,
            block=BasicBlock,
            num_classes=10
        ).to(device)
        
        if mode != 'eager':
            model = torch.compile(model, mode=mode, backend="inductor")
            
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        trainer = Trainer(model, criterion, optimizer, device)
        
        # Track first epoch and steady state performance
        epoch_times = []
        
        for epoch in range(num_epochs):
            torch.cuda.synchronize()
            start_time = time.time()
            
            loss, acc, timing = trainer.train_epoch(
                train_loader,
                epoch,
                measure_time=True
            )
            
            torch.cuda.synchronize()
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)
            
            if epoch == 0:
                print(f"First epoch time: {epoch_time:.2f}s")
            elif epoch >= 5:
                print(f"Epoch {epoch + 1} time: {epoch_time:.2f}s")
        
        results[mode] = {
            'first_epoch_time': epoch_times[0],
            'steady_state_time': sum(epoch_times[5:]) / 5,
            'all_epoch_times': epoch_times,
            'final_accuracy': acc
        }
    
    return results 