import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tinyimagenet import TinyImageNet
from pathlib import Path

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        # Initialize the dataset once
        self.dataset = TinyImageNet(self.root_dir, split=split)
        self.classes = self.dataset.classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.length = len(self.dataset)
        
        # Pre-transform and cache all images and labels
        self._cache = []
        for img, lbl in self.dataset:
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            if self.transform:
                img = self.transform(img)
            self._cache.append((img, lbl))
        
        # Clear the dataset to free memory
        self.dataset = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self._cache[idx]

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

def get_cifar10_dataloaders(root_dir, batch_size=128, num_workers=0, pin_memory=True, persistent_workers=False):
    print("Loading CIFAR-10 dataset...")
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2023, 0.1994, 0.2010])
    ])

    # Just normalization for validation
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2023, 0.1994, 0.2010])
    ])

    # Create full datasets first
    full_train_dataset = datasets.CIFAR10(
        root=root_dir, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=root_dir, 
        train=False, 
        download=True, 
        transform=val_transform
    )

    # Randomly sample 10k images from training set
    print("Sampling 10k images from training set...")
    indices = torch.randperm(len(full_train_dataset))[:10000]
    train_subset = torch.utils.data.Subset(full_train_dataset, indices)
    print(f"Using {len(train_subset)} training images")

    # Split the 10k subset into train and val
    train_size = int(0.9 * len(train_subset))
    val_size = len(train_subset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_subset, [train_size, val_size]
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

    return train_loader, val_loader, test_loader, full_train_dataset.classes

def get_tinyimagenet_dataloaders(root_dir, batch_size=128, num_workers=0, pin_memory=True, persistent_workers=False):
    print("Loading and caching TinyImageNet dataset...")
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Just normalization for validation
    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create full datasets first
    full_train_dataset = TinyImageNetDataset(root_dir, split='train', transform=train_transform)
    val_dataset = TinyImageNetDataset(root_dir, split='val', transform=val_transform)
    test_dataset = TinyImageNetDataset(root_dir, split='test', transform=val_transform)
    
    # Randomly sample 10k images from training set
    print("Sampling 10k images from training set...")
    indices = torch.randperm(len(full_train_dataset))[:10000]
    train_dataset = torch.utils.data.Subset(full_train_dataset, indices)
    print(f"Using {len(train_dataset)} training images")
    print("Dataset loaded and cached!")

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

    return train_loader, val_loader, test_loader, full_train_dataset.classes

def get_dataloaders(dataset_name, root_dir, batch_size=128, num_workers=0, pin_memory=True, persistent_workers=False):
    if dataset_name == 'cifar10':
        return get_cifar10_dataloaders(root_dir, batch_size, num_workers, pin_memory, persistent_workers)
    elif dataset_name == 'tinyimagenet':
        return get_tinyimagenet_dataloaders(root_dir, batch_size, num_workers, pin_memory, persistent_workers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}") 