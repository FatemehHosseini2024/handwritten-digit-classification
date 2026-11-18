import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision import datasets, transforms

DATA_DIR = "data"


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


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

# Flatten images and show shapes before/after
for name, loader in [("Train", train_loader), ("Validation", val_loader), ("Test", test_loader)]:
    images, labels = next(iter(loader))
    print(f"\n{name} batch:")
    print(f"  Before flatten: {images.shape}")
    flattened = images.view(images.size(0), -1)
    print(f"  After flatten:  {flattened.shape}")

# Neural network model
model = MNISTNet()
print(f"\nModel:\n{model}")

# Parameter info
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
print("\nParameter names and shapes:")
for name, param in model.named_parameters():
    print(f"  {name}: {tuple(param.shape)}")

# Forward pass on a training batch
images, labels = next(iter(train_loader))
flattened = images.view(images.size(0), -1)
input_t = flattened.t()
output = model(flattened)
output_t = output.t()

print(f"\nForward pass on train batch:")
print(f"  Input shape:  {tuple(input_t.shape)}")
print(f"  Output shape: {tuple(output_t.shape)}")
print(f"\nOutput for first image (10 logits):")
print(output[0])
print(f"Number of logits: {output[0].numel()}")

# Predicted vs actual label for first image
predicted_label = output[0].argmax().item()
actual_label = labels[0].item()
print(f"\nPredicted label (argmax): {predicted_label}")
print(f"Actual label: {actual_label}")

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
