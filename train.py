import argparse
import torch
import torch.nn as nn
from models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
                              efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import resnet50, resnet152, densenet201
from data.dataset import get_dataloaders
from utils.train import train_model, create_optimizer, create_scheduler, evaluate_model

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

def main():
    parser = argparse.ArgumentParser(description='Train models on image datasets')
    parser.add_argument('--model', type=str, required=True,
                      choices=['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
                              'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                              'efficientnet-b6', 'efficientnet-b7', 'resnet50',
                              'resnet152', 'densenet201'],
                      help='Model to train')
    parser.add_argument('--dataset', type=str, default='tinyimagenet',
                      choices=['tinyimagenet', 'cifar10'],
                      help='Dataset to use for training')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory to store dataset')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for training (larger = faster but more memory)')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='Number of workers for data loading (0 recommended for MPS)')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Directory to save model checkpoints')
    args = parser.parse_args()

    # Set device and MPS optimizations
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        # Enable memory efficient attention if available
        if hasattr(torch.backends.mps, 'is_mem_efficient_attention_enabled'):
            torch.backends.mps.enable_mem_efficient_attention()
        # Set memory format to channels_last for better MPS performance
        torch.backends.mps.set_memory_format(torch.channels_last)
        print('MPS optimizations enabled')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Get dataloaders with optimized settings
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.dataset,
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=0,  # Keep at 0 for MPS
        pin_memory=True,  # Enable pin memory for faster data transfer
        persistent_workers=False  # Disable persistent workers for MPS
    )
    print(f'Number of classes: {len(classes)}')

    # Set input size based on dataset
    input_size = 32 if args.dataset == 'cifar10' else 64

    # Create model
    model = get_model(args.model, len(classes))
    model = model.to(device)
    
    # Convert model to channels_last memory format for MPS
    if torch.backends.mps.is_available():
        model = model.to(memory_format=torch.channels_last)

    # Create loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = create_scheduler(optimizer, args.num_epochs)

    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        args.num_epochs, device, args.save_dir)

    # Evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    print(f'Final Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main() 