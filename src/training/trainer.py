import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, Any
from torch.profiler import profile, record_function, ProfilerActivity

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        enable_profiling: bool = False
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.enable_profiling = enable_profiling
        
        if enable_profiling:
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )

    def train_epoch(
        self, 
        train_loader: DataLoader, 
        epoch: int,
        measure_time: bool = False
    ) -> Tuple[float, float, Dict[str, float]]:
        """Trains the model for one epoch and returns loss, accuracy and timing info"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        timing_info = {
            "data_loading": 0.0,
            "training": 0.0,
            "total": 0.0
        }

        if measure_time:
            epoch_start = time.perf_counter()
            
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", unit="batch") as pbar:
            for batch_idx, (data, target) in enumerate(train_loader):
                if measure_time:
                    batch_start = time.perf_counter()
                    
                data, target = data.to(self.device), target.to(self.device)
                
                if measure_time:
                    timing_info["data_loading"] += time.perf_counter() - batch_start
                    train_start = time.perf_counter()
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                if measure_time:
                    timing_info["training"] += time.perf_counter() - train_start
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                pbar.update(1)
                pbar.set_postfix(
                    loss=loss.item(),
                    acc=f"{(correct / total) * 100:.2f}%"
                )
                
                if self.enable_profiling:
                    self.profiler.step()
        
        if measure_time:
            timing_info["total"] = time.perf_counter() - epoch_start
            
        average_loss = total_loss / len(train_loader)
        accuracy = correct / total * 100
        
        return average_loss, accuracy, timing_info

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluates the model on the test set"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        test_loss /= len(test_loader)
        accuracy = correct / total * 100
        
        return test_loss, accuracy

    def train(
        self, 
        train_loader: DataLoader,
        num_epochs: int,
        test_loader: DataLoader = None,
        measure_time: bool = False
    ) -> Dict[str, Any]:
        """Trains the model for specified number of epochs"""
        results = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'timing_info': []
        }
        
        for epoch in range(num_epochs):
            loss, acc, timing = self.train_epoch(train_loader, epoch, measure_time)
            results['train_loss'].append(loss)
            results['train_acc'].append(acc)
            results['timing_info'].append(timing)
            
            if test_loader is not None:
                test_loss, test_acc = self.evaluate(test_loader)
                results['test_loss'].append(test_loss)
                results['test_acc'].append(test_acc)
                print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        return results 