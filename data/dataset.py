import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path

def get_cifar100_dataloaders(root_dir, batch_size=128, num_workers=0, pin_memory=True, persistent_workers=False):
    print("Loading CIFAR-100 dataset...")
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761])
    ])

    # Just normalization for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                        std=[0.2675, 0.2565, 0.2761])
    ])

    # Create datasets
    train_dataset = datasets.CIFAR100(
        root=root_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=root_dir, 
        train=False, 
        download=True, 
        transform=val_transform
    )

    # Split train into train and val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader, test_loader, train_dataset.dataset.classes

def get_dataloaders(dataset_name, root_dir, batch_size=128, num_workers=0, pin_memory=True, persistent_workers=False):
    if dataset_name == 'cifar100':
        return get_cifar100_dataloaders(root_dir, batch_size, num_workers, pin_memory, persistent_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}") 