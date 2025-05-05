import argparse
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np
from itertools import product
import json
import os
from datetime import datetime
from models.efficientnet import efficientnet_b0
from data.dataset import get_dataloaders
from utils.train import train_model, create_optimizer, create_scheduler, evaluate_model

def calculate_model_size(model):
    """Calculate the number of parameters in the model"""
    return sum(p.numel() for p in model.parameters())

def calculate_flops(model, input_size):
    """Calculate approximate FLOPs for the model"""
    flops = 0
    input_h = input_w = input_size
    
    def conv_flops(module, h, w):
        # For conv layers: (Cin * K^2) * H * W * Cout
        # K is kernel size, H and W are output spatial dimensions
        out_h = (h + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
        out_w = (w + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
        return module.in_channels * module.kernel_size[0] * module.kernel_size[1] * out_h * out_w * module.out_channels

    def linear_flops(module):
        # For linear layers: Cin * Cout
        return module.in_features * module.out_features

    # Track spatial dimensions through the network
    h, w = input_h, input_w
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            flops += conv_flops(module, h, w)
            # Update spatial dimensions for next layer
            h = (h + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) // module.stride[0] + 1
            w = (w + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) // module.stride[1] + 1
        elif isinstance(module, nn.Linear):
            flops += linear_flops(module)
    
    return flops

def create_scaled_model(alpha, beta, gamma, num_classes, input_size):
    """Create an EfficientNet model with the given scaling coefficients"""
    # Calculate scaled input size
    scaled_input_size = int(input_size * gamma)
    
    # Start with EfficientNet-B0 as the base model
    model = efficientnet_b0(num_classes, input_size=scaled_input_size)
    
    # Apply compound scaling
    # Note: This is a simplified version - in practice, you'd need to modify
    # the model architecture to properly apply these scaling factors
    # This is just a placeholder to demonstrate the concept
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Scale number of channels
            module.out_channels = int(module.out_channels * beta)
            module.in_channels = int(module.in_channels * beta)
        elif isinstance(module, nn.Linear):
            # Scale number of features
            module.out_features = int(module.out_features * beta)
            module.in_features = int(module.in_features * beta)
    
    return model

def generate_parameter_combinations():
    """Generate parameter combinations that satisfy α * β² * γ² ≈ 2"""
    # Define ranges for each parameter
    alpha_range = np.linspace(1.0, 1.3, 10)  # Network depth
    beta_range = np.linspace(1.0, 1.3, 10)   # Network width
    gamma_range = np.linspace(1.0, 1.3, 10)  # Resolution
    
    valid_combinations = []
    
    # Generate all possible combinations
    for alpha in alpha_range:
        for beta in beta_range:
            for gamma in gamma_range:
                # Calculate the compound scaling factor
                compound_factor = alpha * (beta ** 2) * (gamma ** 2)
                
                # Check if the combination satisfies the constraint
                if 1.9 <= compound_factor <= 2.1:
                    valid_combinations.append((alpha, beta, gamma))
    
    return valid_combinations

def main():
    parser = argparse.ArgumentParser(description='Search for optimal EfficientNet scaling parameters')
    parser.add_argument('--data_dir', type=str, default='data',
                    help='Directory to store dataset')
    parser.add_argument('--batch_size', type=int, default=512,  # Increased for A100
                    help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                    help='Number of epochs to train each model')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='results/parameter_search',
                    help='Directory to save results')
    parser.add_argument('--max_flops', type=float, default=1e10,
                    help='Maximum allowed FLOPs')
    parser.add_argument('--mixed_precision', action='store_true',
                    help='Use mixed precision training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help='Number of gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true',
                    help='Use Automatic Mixed Precision')
    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device and enable cuDNN optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking
        torch.backends.cudnn.deterministic = False  # Disable deterministic mode for better performance
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('Using CPU')

    # Get dataloaders with optimized settings for A100
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        'cifar100',
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,  # Enable pin memory for faster data transfer to GPU
        persistent_workers=True  # Enable persistent workers for better performance
    )

    # Generate valid parameter combinations
    parameter_combinations = generate_parameter_combinations()
    print(f'Found {len(parameter_combinations)} valid parameter combinations')
    
    results = []
    
    # Grid search
    for alpha, beta, gamma in parameter_combinations:
        print(f'\nTesting parameters: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}')
        print(f'Compound factor: {alpha * (beta ** 2) * (gamma ** 2):.3f}')
        
        # Create model with current parameters
        model = create_scaled_model(alpha, beta, gamma, len(classes), 32)  # CIFAR-100 has 32x32 images
        model = model.to(device)
        
        # Enable DataParallel if multiple GPUs are available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        
        # Calculate model size and FLOPs
        model_size = calculate_model_size(model)
        flops = calculate_flops(model, 32)  # Base input size is 32
        
        print(f'Model size: {model_size:,} parameters')
        print(f'FLOPs: {flops:,.0f}')
        
        # Skip if FLOPs exceed maximum
        if flops > args.max_flops:
            print('Skipping - FLOPs exceed maximum')
            continue
        
        # Create optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
        scheduler = create_scheduler(optimizer, args.num_epochs)
        
        # Create mixed precision scaler if enabled
        scaler = amp.GradScaler() if args.mixed_precision else None
        
        # Train model
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            args.num_epochs, device, args.save_dir, scaler)
        
        # Evaluate on test set
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        
        # Store results
        result = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'compound_factor': alpha * (beta ** 2) * (gamma ** 2),
            'model_size': model_size,
            'flops': flops,
            'test_accuracy': test_acc,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        results.append(result)
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['test_accuracy'])
    print('\nBest parameters:')
    print(f'Alpha: {best_result["alpha"]:.3f}')
    print(f'Beta: {best_result["beta"]:.3f}')
    print(f'Gamma: {best_result["gamma"]:.3f}')
    print(f'Compound factor: {best_result["compound_factor"]:.3f}')
    print(f'Test Accuracy: {best_result["test_accuracy"]:.4f}')
    print(f'Model Size: {best_result["model_size"]:,} parameters')
    print(f'FLOPs: {best_result["flops"]:,.0f}')

if __name__ == '__main__':
    main() 