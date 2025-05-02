import argparse
import torch
import torch.nn as nn
from models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
                              efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import resnet50, resnet152, densenet201
from data.dataset import get_dataloaders
from utils.train import train_model, create_optimizer, create_scheduler, evaluate_model
import torch.optim as optim
import math
from torch.cuda.amp import GradScaler, autocast

def get_model(model_name, num_classes=200, input_size=64):
    if model_name == 'efficientnet-b0':
        return efficientnet_b0(num_classes, input_size=input_size)
    elif model_name == 'efficientnet-b1':
        return efficientnet_b1(num_classes, input_size=input_size)
    elif model_name == 'efficientnet-b2':
        return efficientnet_b2(num_classes, input_size=input_size)
    elif model_name == 'efficientnet-b3':
        return efficientnet_b3(num_classes, input_size=input_size)
    elif model_name == 'efficientnet-b4':
        return efficientnet_b4(num_classes, input_size=input_size)
    elif model_name == 'efficientnet-b5':
        return efficientnet_b5(num_classes, input_size=input_size)
    elif model_name == 'efficientnet-b6':
        return efficientnet_b6(num_classes, input_size=input_size)
    elif model_name == 'efficientnet-b7':
        return efficientnet_b7(num_classes, input_size=input_size)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'resnet152':
        model = resnet152(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'densenet201':
        model = densenet201(pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_optimal_params(model_name):
    """
    Return optimal parameters for each model based on the EfficientNet paper and standard practices
    """
    if model_name.startswith('efficientnet'):
        # EfficientNet parameters from paper
        return {
            'batch_size': 2048,  # Paper uses 2048 but we'll adjust based on GPU
            'learning_rate': 0.016,  # Base learning rate
            'weight_decay': 1e-5,
            'optimizer': 'rmsprop',
            'momentum': 0.9,
            'alpha': 0.9,
            'eps': 0.001,
            'num_epochs': 350,  # Paper uses 350 epochs
            'scheduler': 'cosine',
            'warmup_epochs': 5
        }
    elif model_name == 'resnet50' or model_name == 'resnet152':
        # ResNet parameters from paper
        return {
            'batch_size': 256,
            'learning_rate': 0.1,
            'weight_decay': 1e-4,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'num_epochs': 200,
            'scheduler': 'step',  # Original paper uses step decay
            'step_size': 30,  # Divide learning rate by 10 every 30 epochs
            'gamma': 0.1,  # Learning rate decay factor
            'warmup_epochs': 0  # No warmup in original paper
        }
    elif model_name == 'densenet201':
        # DenseNet parameters from paper
        return {
            'batch_size': 64,  # Original paper uses 64
            'learning_rate': 0.1,
            'weight_decay': 1e-4,
            'optimizer': 'sgd',
            'momentum': 0.9,
            'num_epochs': 300,
            'scheduler': 'step',  # Original paper uses step decay
            'step_size': 30,  # Divide learning rate by 10 every 30 epochs
            'gamma': 0.1,  # Learning rate decay factor
            'warmup_epochs': 0  # No warmup in original paper
        }

def main():
    parser = argparse.ArgumentParser(description='Train models on image datasets')
    parser.add_argument('--model', type=str, required=True,
                      choices=['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
                              'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                              'efficientnet-b6', 'efficientnet-b7', 'resnet50',
                              'resnet152', 'densenet201'],
                      help='Model to train')
    parser.add_argument('--dataset', type=str, default='tinyimagenet',
                      choices=['tinyimagenet', 'cifar10', 'cifar100'],
                      help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory to store dataset')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Batch size for training (if None, uses optimal for model)')
    parser.add_argument('--num_epochs', type=int, default=None,
                      help='Number of epochs to train (if None, uses optimal for model)')
    parser.add_argument('--learning_rate', type=float, default=None,
                      help='Initial learning rate (if None, uses optimal for model)')
    parser.add_argument('--weight_decay', type=float, default=None,
                      help='Weight decay (if None, uses optimal for model)')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Directory to save model checkpoints')
    args = parser.parse_args()

    # Get optimal parameters for the model
    params = get_optimal_params(args.model)
    
    # Override with command line arguments if provided
    if args.batch_size is not None:
        params['batch_size'] = args.batch_size
    if args.num_epochs is not None:
        params['num_epochs'] = args.num_epochs
    if args.learning_rate is not None:
        params['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        params['weight_decay'] = args.weight_decay

    # Set device to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Enable cuDNN benchmarking for better performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
        print(f'Compute Capability: {torch.cuda.get_device_capability(0)}')
        
        # Enable TF32 for A100
        if torch.cuda.get_device_capability(0)[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print('TF32 enabled for A100')

    # Get dataloaders with CUDA-optimized settings
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.dataset,
        args.data_dir, 
        batch_size=params['batch_size'], 
        num_workers=args.num_workers,
        pin_memory=True,  # Enable pin memory for faster data transfer to GPU
        persistent_workers=True  # Keep workers alive between epochs
    )
    print(f'Number of classes: {len(classes)}')

    # Set input size based on dataset
    input_size = 32 if args.dataset in ['cifar10', 'cifar100'] else 64

    # Create model and move to GPU
    model = get_model(args.model, len(classes))
    model = model.to(device)
    
    # Enable memory efficient attention if available
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_attention'):
        torch.backends.cuda.enable_mem_efficient_attention()

    # Create loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        params['num_epochs'], device, args.save_dir)

    # Evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    print(f'Final Test Accuracy: {test_acc:.4f}')

def create_optimizer(model, params):
    """
    Create optimizer based on model parameters
    """
    if params['optimizer'] == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=params['learning_rate'],
            alpha=params['alpha'],
            momentum=params['momentum'],
            eps=params['eps'],
            weight_decay=params['weight_decay']
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            momentum=params['momentum'],
            weight_decay=params['weight_decay'],
            nesterov=True
        )
    return optimizer

def create_scheduler(optimizer, params):
    """
    Create learning rate scheduler based on model parameters
    """
    if params['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params['num_epochs'] - params['warmup_epochs'],
            eta_min=params['learning_rate'] * 0.01
        )
        if params['warmup_epochs'] > 0:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=params['warmup_epochs']
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[params['warmup_epochs']]
            )
    elif params['scheduler'] == 'step':
        # Step decay scheduler used in ResNet and DenseNet papers
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params['step_size'],
            gamma=params['gamma']
        )
    return scheduler

if __name__ == '__main__':
    main() 