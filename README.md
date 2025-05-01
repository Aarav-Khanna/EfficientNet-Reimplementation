# EfficientNet Reimplementation

This project implements the EfficientNet architecture as described in the paper ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946) by Mingxing Tan and Quoc V. Le.

## Features

- Implementation of EfficientNet B0-B7 variants
- Support for both CIFAR-10 and TinyImageNet datasets
- Training utilities with learning rate scheduling
- Model analysis and visualization tools
- MPS (Metal Performance Shaders) support for M1 Macs

## Project Structure

```
EfficientNet-Reimplementation/
├── data/
│   ├── dataset.py          # Dataset loading and preprocessing
│   └── __init__.py
├── models/
│   ├── efficientnet.py     # EfficientNet implementation
│   └── __init__.py
├── utils/
│   ├── train.py           # Training utilities
│   └── __init__.py
├── analysis/
│   ├── plot_results.py    # Visualization tools
│   └── __init__.py
├── train.py               # Main training script
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EfficientNet-Reimplementation.git
cd EfficientNet-Reimplementation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a model on CIFAR-10 (recommended for testing):
```bash
python train.py --model efficientnet-b0 --dataset cifar10 --batch_size 128 --num_epochs 100
```

Train a model on TinyImageNet (full training):
```bash
python train.py --model efficientnet-b0 --dataset tinyimagenet --batch_size 128 --num_epochs 350
```

### Available Models

- EfficientNet variants: `efficientnet-b0` through `efficientnet-b7`
- Reference models: `resnet50`, `resnet152`, `densenet201`

### Training Parameters

- `--model`: Model architecture to train
- `--dataset`: Dataset to use (`cifar10` or `tinyimagenet`)
- `--batch_size`: Batch size for training (default: 128)
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Initial learning rate (default: 0.1)
- `--weight_decay`: Weight decay (default: 1e-4)
- `--data_dir`: Directory to store datasets (default: 'data')
- `--save_dir`: Directory to save model checkpoints (default: 'results')

## Implementation Details

### EfficientNet Architecture

The implementation follows the paper's architecture:
- MBConv blocks with squeeze-and-excitation
- Swish activation function
- Proper width and depth scaling for B0-B7 variants
- Input size: 224x224 (standard) or 64x64 (for CIFAR-10/TinyImageNet)

### Dataset Handling

- CIFAR-10: 32x32 images, 10 classes
- TinyImageNet: 64x64 images, 200 classes
- Automatic data augmentation and normalization
- Efficient data loading with caching

## Results

The implementation aims to reproduce the paper's results:
- EfficientNet-B0: ~5.3M parameters
- EfficientNet-B1: ~7.8M parameters
- EfficientNet-B2: ~9.2M parameters
- EfficientNet-B3: ~12M parameters
- EfficientNet-B4: ~19M parameters
- EfficientNet-B5: ~30M parameters
- EfficientNet-B6: ~43M parameters
- EfficientNet-B7: ~66M parameters

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original paper: ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946)
- PyTorch implementation reference 