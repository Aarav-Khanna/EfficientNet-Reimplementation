# EfficientNet Reimplementation

## Introduction

This repository contains a re-implementation of the paper ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](https://arxiv.org/abs/1905.11946) by Mingxing Tan and Quoc V. Le. The main contribution of the paper is the introduction of a compound scaling method for CNNs, enabling efficient scaling of depth, width, and resolution to achieve state-of-the-art accuracy with fewer resources. This project adapts EfficientNet for CIFAR100 classification and provides tools for model analysis and interpretability.

## Chosen Result

We reimplemented the core contributions of the EfficientNet paper. First, we reproduced the grid search for the optimal compound scaling coefficients (α, β, γ) under the constraint α * β² * γ² ≈ 2, as described in the original work. Using these coefficients, we constructed EfficientNet models B0 through B7 and trained them on CIFAR-100. We then replicated the key result of the paper: the accuracy improvements as model size increases, comparing our EfficientNet models to ResNet-50, DenseNet-201, and ResNet-152 on CIFAR-100, all under the same computational constraints—mirroring the analysis in Figure 1 of the original paper. Additionally, we reproduced the validation and test accuracy results (Table 8 in the paper) for our models. Finally, we generated Class Activation Map (CAM) visualizations to replicate the interpretability analysis shown in Figure 7 of the paper.

## GitHub Contents

```
EfficientNet-Reimplementation/
├── code/
│   ├── data/                # CIFAR100 dataset loading and preprocessing
│   ├── models/              # EfficientNet implementation
│   ├── utils/               # Training utilities
│   ├── train.py             # Main training script
│   ├── parameter_search.py  # Compound scaling parameter search
│   └── cam_visualization_colab.py  # CAM visualization
├── results/                 # Plots and visualizations
├── README.md                # Project overview and instructions
├── requirements.txt         # Dependencies
└── LICENSE                  # License file
```

- All code is under `code/`.
- Results plots and visualizations are in `results/`.
- Dataset handling scripts are in `code/data/`.

## Re-implementation Details

- **Approach:** Faithful re-implementation of EfficientNet (B0-B7) for CIFAR100, including compound scaling and model interpretability tools.
- **Models:** EfficientNet B0-B7, with reference models (ResNet, DenseNet) for comparison.
- **Datasets:** CIFAR100 (32x32 images, 100 classes), with advanced augmentation.
- **Tools:** PyTorch, mixed precision (AMP), multi-GPU support, parameter search for scaling.
- **Evaluation:** Accuracy, model size, FLOPs, and CAM-based interpretability.
- **Challenges:** Adapting EfficientNet to CIFAR100's small input size, optimizing for A100 GPUs, and automating compound scaling search.

## Reproduction Steps

To reproduce the results or use this repo:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/EfficientNet-Reimplementation.git
   cd EfficientNet-Reimplementation
   ```
2. **Install dependencies:**
   ```bash
   pip install -r code/requirements.txt
   ```
3. **Download CIFAR100:**
   The dataset will be automatically downloaded to `data/` when running the training script.
4. **Train a model:**
   ```bash
   python code/train.py --model efficientnet-b0 --batch_size 256 --num_epochs 200
   ```
   For mixed precision (recommended for A100):
   ```bash
   python code/train.py --model efficientnet-b0 --mixed_precision
   ```
5. **Parameter search:**
   ```bash
   python code/parameter_search.py --mixed_precision
   ```
6. **CAM visualization:**
   ```bash
   python code/cam_visualization_colab.py --images_path /path/to/images --output_path /path/to/output
   ```

- **Dependencies:** See `code/requirements.txt` for required libraries.
- **Resources:** A100 GPU recommended for fastest training; multi-GPU supported.

## Results/Insights

- Our EfficientNet re-implementation achieves competitive accuracy on CIFAR100, closely matching the scaling trends reported in the original paper.
- Compound scaling parameter search finds optimal depth/width/resolution combinations under FLOPs constraints.
- CAM visualizations show that compound-scaled models focus more accurately on relevant image regions, supporting the paper's claims.
- See `results/` for accuracy plots and CAM visualizations.

## Conclusion

This re-implementation validates the effectiveness of compound scaling for efficient model design. The codebase provides a reproducible framework for further research on model scaling and interpretability.

## References

- Tan, M., & Le, Q. V. (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946). arXiv preprint arXiv:1905.11946.
- PyTorch documentation: https://pytorch.org/
- CIFAR100 dataset: https://www.cs.toronto.edu/~kriz/cifar.html

## Acknowledgements

- Original authors of EfficientNet for their foundational work.
- PyTorch community for open-source tools and documentation.
- This re-implementation was completed as part of a course project for CS4782, adding peer-reviewed authenticity. 