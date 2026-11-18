import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from data import get_datasets, get_dataloaders
from model import MNISTNet
from train import train
from evaluate import evaluate, print_metrics
from preprocess import load_image, preprocess
from predict import predict

# --- Data ---
train_set, test_set, train_subset, val_subset = get_datasets()
train_loader, val_loader, test_loader = get_dataloaders(train_subset, val_subset, test_set)

print(f"Train set: {len(train_set)} samples")
print(f"Test set: {len(test_set)} samples")
print(f"Train subset: {len(train_subset)} samples")
print(f"Validation subset: {len(val_subset)} samples")
print(f"Train loader batches: {len(train_loader)}")
print(f"Validation loader batches: {len(val_loader)}")
print(f"Test loader batches: {len(test_loader)}")

# --- Model ---
model = MNISTNet()
print(f"\nModel:\n{model}")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
print("Parameter names and shapes:")
for name, param in model.named_parameters():
    print(f"  {name}: {tuple(param.shape)}")

# --- Training ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses, val_accuracies, best_val_loss = train(
    model, train_loader, val_loader, criterion, optimizer, epochs=15, patience=3
)

# --- Save best model ---
torch.save(model.state_dict(), "best_mnist_model.pth")
print("Best model saved to best_mnist_model.pth")

# --- Evaluation ---
test_accuracy, all_preds, all_labels = evaluate(model, test_loader)
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

print_metrics(all_labels, all_preds)

# --- Confusion Matrix ---
cm = None
try:
    import numpy as np
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
except Exception as e:
    print(f"Could not compute confusion matrix: {e}")

# --- Plot loss curve ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("Saved loss curve to loss_curve.png")

# --- Confusion matrix plot ---
if cm is not None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - MNIST Test Set')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion matrix to confusion_matrix.png")

# --- External image inference ---
image_path = input("\nEnter the path to your image: ")
try:
    img = load_image(image_path)
    tensor = preprocess(img)
    predicted_digit = predict(tensor)
    print(f"\nFinal Predicted Digit: {predicted_digit}")
except Exception as e:
    print(f"Error processing image: {e}")
