# EfficientNet Reimplementation

This repository contains a reimplementation of the EfficientNet architecture as described in the paper ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/pdf/1905.11946) by Mingxing Tan and Quoc V. Le.

## Project Overview

This project includes:
- Implementation of EfficientNet architecture from scratch
- Training and evaluation on TinyImageNet dataset
- Comparative analysis with other architectures (ResNet-50, ResNet-152, DenseNet-201)
- Reproduction of key results from the original paper

## Requirements

The project requires Python 3.8+ and the following packages:
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- seaborn
- tinyimagenet

See `requirements.txt` for specific versions.

## Project Structure

```
EfficientNet-Reimplementation/
├── models/
│   ├── efficientnet.py      # EfficientNet implementation
│   ├── resnet.py           # ResNet implementation
│   └── densenet.py         # DenseNet implementation
├── data/
│   └── tinyimagenet.py     # TinyImageNet dataset handling
├── utils/
│   ├── train.py           # Training utilities
│   └── evaluate.py        # Evaluation utilities
├── notebooks/
│   └── analysis.ipynb     # Analysis and visualization
├── results/               # Saved model weights and results
├── requirements.txt
└── README.md
```

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and prepare TinyImageNet dataset:
```bash
python data/tinyimagenet.py
```

3. Train models:
```bash
python train.py --model efficientnet-b0
```

4. Evaluate models:
```bash
python evaluate.py --model efficientnet-b0
```

## Results

The project aims to reproduce key results from the original paper:
- Figure 1: Model scaling comparison
- Table 2: Accuracy comparison with other architectures

## References

- [EfficientNet Paper](https://arxiv.org/pdf/1905.11946)
- [Original Implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet) 