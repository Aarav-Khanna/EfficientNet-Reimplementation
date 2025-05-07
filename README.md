# EfficientNet Reimplementation

This project implements the EfficientNet architecture as described in the paper ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946) by Mingxing Tan and Quoc V. Le, optimized for CIFAR100 classification on NVIDIA A100 GPUs.

## Features

- Implementation of EfficientNet B0-B7 variants
- Optimized for CIFAR100 dataset (100 classes)
- Training utilities with advanced learning rate scheduling
- Model analysis and visualization tools
- NVIDIA A100 GPU optimizations with mixed precision training
- Multi-GPU support with DataParallel
- Advanced data augmentation and preprocessing
- Compound scaling parameter search with constraint optimization
- Resource-aware model scaling
- Class Activation Map (CAM) visualization for model interpretability

## Project Structure

```
EfficientNet-Reimplementation/
├── data/
│   ├── dataset.py          # CIFAR100 dataset loading and preprocessing
│   └── __init__.py
├── models/
│   ├── efficientnet.py     # EfficientNet implementation
│   └── __init__.py
├── utils/
│   ├── train.py           # Training utilities with mixed precision support
│   └── __init__.py
├── analysis/
│   ├── plot_results.py    # Visualization tools
│   └── __init__.py
├── train.py               # Main training script
├── parameter_search.py    # Compound scaling parameter search
├── cam_visualization_colab.py  # CAM visualization in Google Colab
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

Train a model on CIFAR100 with default settings:
```bash
python train.py --model efficientnet-b0 --batch_size 256 --num_epochs 200
```

Train with mixed precision (recommended for A100):
```bash
python train.py --model efficientnet-b0 --mixed_precision
```

### Parameter Search

Search for optimal compound scaling parameters:
```bash
python parameter_search.py --mixed_precision
```

The parameter search implements the compound scaling constraint from the paper:
- α * β² * γ² ≈ 2 (where α is depth, β is width, γ is resolution)
- Searches for combinations where 1.9 ≤ α * β² * γ² ≤ 2.1
- Tracks model size and FLOPs for each combination

### Class Activation Map (CAM) Visualization

Visualize and compare Class Activation Maps across different scaling strategies:
```bash
python cam_visualization_colab.py --images_path /path/to/images --output_path /path/to/output
```

This creates visualizations showing:
- Original images
- Activation heatmaps for baseline model
- Activation heatmaps for deeper models
- Activation heatmaps for wider models
- Activation heatmaps for higher resolution models
- Activation heatmaps for compound scaling models

The CAM visualizations help interpret where the model focuses its attention when making predictions, highlighting the benefit of compound scaling in creating more focused and accurate attention maps.

### Available Models

- EfficientNet variants: `efficientnet-b0` through `efficientnet-b7`
- Reference models: `resnet50`, `resnet152`, `densenet201`

### Training Parameters

- `--model`: Model architecture to train
- `--batch_size`: Batch size for training (default: 256)
- `--num_epochs`: Number of training epochs (default: 200)
- `--learning_rate`: Initial learning rate (default: 0.1)
- `--weight_decay`: Weight decay (default: 5e-4)
- `--num_workers`: Number of data loading workers (default: 8)
- `--data_dir`: Directory to store dataset (default: 'data')
- `--save_dir`: Directory to save model checkpoints (default: 'results')
- `--mixed_precision`: Enable mixed precision training (recommended for A100)
- `--max_flops`: Maximum allowed FLOPs for parameter search (default: 1e10)

## Implementation Details

### EfficientNet Architecture

The implementation follows the paper's architecture:
- MBConv blocks with squeeze-and-excitation
- Swish activation function
- Proper width and depth scaling for B0-B7 variants
- Input size: 32x32 (optimized for CIFAR100)

### Compound Scaling

The implementation includes the paper's compound scaling method:
- α: Network depth scaling factor
- β: Network width scaling factor
- γ: Resolution scaling factor
- Constraint: α * β² * γ² ≈ 2
- Parameter search to find optimal combinations
- Resource-aware scaling with FLOPs constraints

### GPU Optimizations

- Automatic Mixed Precision (AMP) training
- Multi-GPU support with DataParallel
- Optimized data loading with pin memory
- Non-blocking data transfers
- cuDNN benchmarking enabled
- Gradient clipping for stability

### Dataset Handling

- CIFAR100: 32x32 images, 100 classes
- Advanced data augmentation:
  - Random cropping with padding
  - Random horizontal flipping
  - Random rotation
  - Color jittering
- Efficient data loading with persistent workers
- Automatic dataset download and caching

### Model Interpretability

The project includes tools for model interpretability:
- Class Activation Map (CAM) visualization for all scaling approaches
- Comparison of attention patterns across different scaling methods
- Custom colormap similar to the one used in the EfficientNet paper
- Integration with Google Colab for easy visualization sharing
- Support for batch processing multiple images

## Results

The implementation aims to achieve state-of-the-art results on CIFAR100:
- EfficientNet-B0: ~5.3M parameters
- EfficientNet-B1: ~7.8M parameters
- EfficientNet-B2: ~9.2M parameters
- EfficientNet-B3: ~12M parameters
- EfficientNet-B4: ~19M parameters
- EfficientNet-B5: ~30M parameters
- EfficientNet-B6: ~43M parameters
- EfficientNet-B7: ~66M parameters

The parameter search helps find optimal scaling combinations that:
- Maximize model accuracy
- Stay within computational budget
- Satisfy the compound scaling constraint
- Balance depth, width, and resolution

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original paper: ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946)
- PyTorch implementation reference 