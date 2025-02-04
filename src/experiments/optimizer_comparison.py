import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Callable
from ..models.resnet import ResNet, BasicBlock
from ..training.trainer import Trainer

def get_optimizer_configs() -> Dict[str, Callable]:
    """Returns a dictionary of optimizer configurations"""
    return {
        'SGD': lambda params: optim.SGD(
            params, 
            lr=0.1, 
            momentum=0.9, 
            weight_decay=5e-4
        ),
        'SGD_Nesterov': lambda params: optim.SGD(
            params, 
            lr=0.1, 
            momentum=0.9, 
            weight_decay=5e-4, 
            nesterov=True
        ),
        'Adagrad': lambda params: optim.Adagrad(
            params, 
            lr=0.1, 
            weight_decay=5e-4
        ),
        'Adadelta': lambda params: optim.Adadelta(
            params, 
            lr=0.1, 
            weight_decay=5e-4
        ),
        'Adam': lambda params: optim.Adam(
            params, 
            lr=0.1, 
            weight_decay=5e-4
        )
    }

def compare_optimizers(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5
) -> Dict[str, Any]:
    """Compares different optimizers' performance"""
    optimizer_configs = get_optimizer_configs()
    results = {}
    
    for opt_name, opt_func in optimizer_configs.items():
        print(f"\nTraining with {opt_name}")
        
        model = ResNet(
            img_channels=3,
            num_layers=18,
            block=BasicBlock,
            num_classes=10
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = opt_func(model.parameters())
        
        trainer = Trainer(model, criterion, optimizer, device)
        opt_results = trainer.train(
            train_loader,
            num_epochs,
            test_loader,
            measure_time=True
        )
        
        results[opt_name] = opt_results
    
    return results 