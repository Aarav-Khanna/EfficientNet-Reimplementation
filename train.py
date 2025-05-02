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
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Directory to save model checkpoints')
    parser.add_argument('--resume', action='store_true',
                      help='Resume training from checkpoint')
    args = parser.parse_args()

    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Get dataloaders with optimized settings
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.dataset,  # Add dataset name as first argument
        args.data_dir, 
        batch_size=args.batch_size, 
        num_workers=0,  # Set to 0 for M1 Mac
        pin_memory=True,  # Enable pin memory for faster data transfer
        persistent_workers=False  # Disable persistent workers for M1
    )
    print(f'Number of classes: {len(classes)}')

    # Set input size based on dataset
    input_size = 32 if args.dataset == 'cifar10' else 64

    # Create model
    model = get_model(args.model, len(classes))
    model = model.to(device)

    # Create loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = create_scheduler(optimizer, args.num_epochs)

    # Load checkpoint if resuming
    start_epoch = 0
    best_acc = 0.0
    # if args.resume:
    #     start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, args.save_dir)
    #     print(f"Resuming training from epoch {start_epoch} with best accuracy {best_acc:.4f}")

    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        args.num_epochs, device, args.save_dir, start_epoch=start_epoch, best_acc=best_acc)

    # Evaluate on test set
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    print(f'Final Test Accuracy: {test_acc:.4f}')

if __name__ == '__main__':
    main() 