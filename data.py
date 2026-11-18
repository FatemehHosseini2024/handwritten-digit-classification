import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

DATA_DIR = "data"
BATCH_SIZE = 64
VAL_SIZE = 5000

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def get_datasets():
    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    train_size = len(train_set) - VAL_SIZE
    train_subset, val_subset = random_split(train_set, [train_size, VAL_SIZE])
    return train_set, test_set, train_subset, val_subset


def get_dataloaders(train_subset, val_subset, test_set):
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader
