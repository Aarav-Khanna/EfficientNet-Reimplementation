import argparse
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, \
                            efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import resnet50, resnet152, densenet201
from data.dataset import get_dataloaders
from utils.train import train_model, create_optimizer, create_scheduler, evaluate_model

def get_model(model_name, num_classes=100, input_size=32):
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
    parser = argparse.ArgumentParser(description='Train models on CIFAR100')
    parser.add_argument('--model', type=str, required=True,
                    choices=['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
                            'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                            'efficientnet-b6', 'efficientnet-b7', 'resnet50',
                            'resnet152', 'densenet201'],
                    help='Model to train')
    parser.add_argument('--data_dir', type=str, default='data',
                    help='Directory to store dataset')
    parser.add_argument('--batch_size', type=int, default=256,  # Increased for A100
                    help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=200,
                    help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8,  # Increased for A100
                    help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='results',
                    help='Directory to save model checkpoints')
    parser.add_argument('--mixed_precision', action='store_true',
                    help='Use mixed precision training')
    args = parser.parse_args()

    # Set device
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
    print(f'Number of classes: {len(classes)}')

    # Create model
    model = get_model(args.model, len(classes))
    model = model.to(device)
    
    # Enable DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # Create loss function, optimizer, and scheduler
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

    print(f'Final Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main() 