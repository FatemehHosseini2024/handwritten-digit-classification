import torch
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision import datasets, transforms

DATA_DIR = "data"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

# Train / validation split (random)
val_size = 5000
train_size = len(train_set) - val_size
train_subset, val_subset = random_split(train_set, [train_size, val_size])

print(f"Train set: {len(train_set)} samples")
print(f"Test set: {len(test_set)} samples")
print(f"Train subset: {len(train_subset)} samples")
print(f"Validation subset: {len(val_subset)} samples")

# DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train loader batches: {len(train_loader)}")
print(f"Validation loader batches: {len(val_loader)}")
print(f"Test loader batches: {len(test_loader)}")

image, label = train_set[0]
print(f"Label: {label}")
print(f"Label type: {type(label)}")
print(f"Image type: {type(image)}")
print(f"Image shape: {image.shape}")
print(f"Image ndim: {image.ndim}")
print(f"Image dtype: {image.dtype}")
print(f"Image min: {image.min().item()}")
print(f"Image max: {image.max().item()}")

import matplotlib.pyplot as plt

# Convert from [1, 28, 28] to [28, 28] for display
img_np = image.squeeze().numpy()
plt.figure(figsize=(4, 4))
plt.imshow(img_np, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.savefig("mnist_first_sample.png", bbox_inches='tight', dpi=150)
print(f"Saved image to mnist_first_sample.png")
