# ResNet Training and Optimization

A comprehensive PyTorch implementation exploring various deep learning optimization techniques using ResNet18 on CIFAR10. This project demonstrates practical approaches to improve training efficiency and performance.

## Features

- **ResNet18 Implementation**
  - Configurable architecture
  - Optional batch normalization
  - CIFAR10 dataset support

- **Optimization Experiments**
  - I/O Pipeline Optimization
  - GPU vs CPU Performance Analysis
  - Optimizer Comparisons
    - SGD with momentum
    - SGD with Nesterov
    - Adagrad
    - Adadelta
    - Adam
  - Batch Normalization Impact Study
  - torch.compile Acceleration

- **Performance Monitoring**
  - Training metrics tracking
  - Profiling support
  - Detailed timing analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/kherihughes/resnet-optimization-suite
cd resnet-optimization-suite

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Usage

### Running Individual Experiments

```bash
# I/O Optimization
python3 scripts/train.py --experiment io --data_path ./data --batch_size 128

# GPU vs CPU Comparison
python3 scripts/train.py --experiment gpu --use_cuda True

# Optimizer Comparison
python3 scripts/train.py --experiment optimizer --use_cuda True

# Compile Acceleration (requires CUDA)
python3 scripts/train.py --experiment compile --use_cuda True
```

### Common Options
```
--experiment    Required    Experiment to run (io/gpu/optimizer/compile)
--use_cuda     Optional    Use CUDA if available (default: True)
--data_path    Optional    Path to CIFAR10 data (default: './data')
--num_workers  Optional    DataLoader workers (default: 2)
--batch_size   Optional    Training batch size (default: 128)
--profile      Optional    Enable profiling (default: False)
```

### Running All Experiments
```bash
# Make the script executable
chmod +x run_all.sh

# Run all experiments
./run_all.sh
```

## Project Structure

```
resnet-optimization-suite/
├── src/
│   ├── models/          # Neural network architectures
│   ├── training/        # Training utilities and core loop
│   └── experiments/     # Individual experiment implementations
├── scripts/             # Entry point scripts
├── requirements.txt     # Project dependencies
└── README.md           # Documentation
```

## Experiment Results

### I/O Optimization
- Analyzes optimal number of DataLoader workers
- Measures data loading vs. computation time
- Finds best configuration for given hardware

### GPU vs CPU Performance
- Comparative training speed analysis
- Memory usage patterns
- Hardware utilization metrics

### Optimizer Comparison
Benchmarks different optimizers:
- Training convergence rates
- Final model accuracy
- Resource utilization

### Batch Normalization Impact
Studies the effect of batch normalization on:
- Training stability
- Convergence speed
- Final model performance

### torch.compile Acceleration
Evaluates different compilation modes:
- First epoch compilation overhead
- Steady-state performance gains
- Memory usage impact

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

[Kheri](https://github.com/kherihughes) 