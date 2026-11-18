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

# Batch accuracy
predictions = output.argmax(dim=1)
correct = (predictions == labels).sum().item()
accuracy = correct / labels.size(0) * 100
print(f"\nBatch accuracy: {correct}/{labels.size(0)} = {accuracy:.2f}%")

# CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
loss = criterion(output, labels)
print(f"\nLoss: {loss}")
print(f"Loss type: {type(loss)}")
print(f"Loss is scalar: {loss.dim() == 0}")
print(f"Loss value positive: {loss.item() > 0}")

# Training step: gradients
model.train()
model.zero_grad()

images_batch, labels_batch = next(iter(train_loader))
flattened_batch = images_batch.view(images_batch.size(0), -1)
output_batch = model(flattened_batch)
loss_batch = criterion(output_batch, labels_batch)
loss_batch.backward()

print(f"\nGradient checks after backward():")
for name, param in model.named_parameters():
    grad = param.grad
    print(f"  {name}.grad is not None: {grad is not None}")
    print(f"  {name}.grad shape: {tuple(grad.shape)} (param shape: {tuple(param.shape)})")
    print(f"  {name}.grad matches param shape: {grad.shape == param.shape}")
    print(f"  {name}.grad sample values: {grad.flatten()[:5].tolist()}")
    print()

# Adam optimizer update
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Capture loss before update
loss_before = loss_batch.item()

# Capture a sample of weights before update
fc1_weight_before = model.fc1.weight.data.clone()
fc3_weight_before = model.fc3.weight.data.clone()

# Zero gradients, forward, loss, backward, step
optimizer.zero_grad()
images_batch, labels_batch = next(iter(train_loader))
flattened_batch = images_batch.view(images_batch.size(0), -1)
output_batch = model(flattened_batch)
loss_batch = criterion(output_batch, labels_batch)
loss_batch.backward()
optimizer.step()

# Capture loss after update
loss_after = loss_batch.item()

# Check if weights changed
fc1_weight_changed = not torch.equal(fc1_weight_before, model.fc1.weight.data)
fc3_weight_changed = not torch.equal(fc3_weight_before, model.fc3.weight.data)

print(f"\nOptimizer weight update:")
print(f"  Loss before update: {loss_before:.6f}")
print(f"  Loss after update:  {loss_after:.6f}")
print(f"  fc1.weight changed: {fc1_weight_changed}")
print(f"  fc3.weight changed: {fc3_weight_changed}")

# Training loop: 15 epochs with early stopping
EPOCHS = 15
PATIENCE = 3
train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    # Training
    model.train()
    epoch_loss = 0.0

    for batch_images, batch_labels in train_loader:
        optimizer.zero_grad()
        flattened_batch = batch_images.view(batch_images.size(0), -1)
        output_batch = model(flattened_batch)
        loss_batch = criterion(output_batch, batch_labels)
        loss_batch.backward()
        optimizer.step()
        epoch_loss += loss_batch.item() * batch_images.size(0)

    avg_train_loss = epoch_loss / len(train_subset)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            flattened_val = val_images.view(val_images.size(0), -1)
            val_output = model(flattened_val)
            v_loss = criterion(val_output, val_labels)
            val_loss += v_loss.item() * val_images.size(0)
            _, predicted = torch.max(val_output, 1)
            correct += (predicted == val_labels).sum().item()
            total += val_labels.size(0)

    avg_val_loss = val_loss / total
    val_accuracy = correct / total * 100
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - "
          f"Train Loss: {avg_train_loss:.6f} - "
          f"Val Loss: {avg_val_loss:.6f} - "
          f"Val Accuracy: {val_accuracy:.2f}%")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  -> Validation loss improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\nBest model restored (best val loss: {best_val_loss:.6f}).")

# Load the saved model
model.load_state_dict(torch.load("best_mnist_model.pth"))
print("Best model loaded from best_mnist_model.pth")

# Evaluate on test set
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for test_images, test_labels in test_loader:
            flattened_test = test_images.view(test_images.size(0), -1)
            test_output = model(flattened_test)
            _, predicted = torch.max(test_output, 1)
            correct += (predicted == test_labels).sum().item()
            total += test_labels.size(0)
    return correct / total * 100

test_accuracy = evaluate(model, test_loader)
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# Collect predictions and true labels for metrics
all_preds = []
all_labels = []
model.eval()
with torch.no_grad():
    for test_images, test_labels in test_loader:
        flattened_test = test_images.view(test_images.size(0), -1)
        test_output = model(flattened_test)
        _, predicted = torch.max(test_output, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())

from sklearn.metrics import classification_report, precision_recall_fscore_support

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4))

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
print("Per-class metrics:")
for i in range(10):
    print(f"  Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

# Macro and Weighted averages
macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

print(f"\nMacro Average: Precision={macro_p:.4f}, Recall={macro_r:.4f}, F1={macro_f:.4f}")
print(f"Weighted Average: Precision={weighted_p:.4f}, Recall={weighted_r:.4f}, F1={weighted_f:.4f}")

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - MNIST Test Set')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion matrix to confusion_matrix.png")

# Plot train and validation loss
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

print(f"\nTraining complete. Recorded {len(train_losses)} epoch losses.")
print(f"Final epoch - Train Loss: {train_losses[-1]:.6f}, "
      f"Val Loss: {val_losses[-1]:.6f}, "
      f"Val Accuracy: {val_accuracies[-1]:.2f}%")

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
from PIL import Image
import os

# Receive an external image from the user
image_path = input("Enter the path to your image: ")

# Check that the file exists and is readable
if not os.path.exists(image_path):
    print(f"Error: File does not exist: {image_path}")
else:
    try:
        img = Image.open(image_path)
        img.load()  # Force load to check readability
    except Exception as e:
        print(f"Error: Cannot read image file: {e}")
    else:
        print(f"Original Image")
        print(f"Image Width and Height: {img.width} x {img.height}")
        print(f"Image Mode: {img.mode}")
        print(f"Number of channels: {len(img.getbands())}")

        # Display the original image
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')
        plt.show()

        # 2.1. Grayscale Conversion
        gray = img.convert('L')
        print(f"\nGrayscale Image")
        print(f"Image Width and Height: {gray.width} x {gray.height}")
        print(f"Image Mode: {gray.mode}")
        print(f"Number of channels: {len(gray.getbands())}")

        # Display grayscale image
        plt.figure(figsize=(4, 4))
        plt.imshow(gray, cmap='gray')
        plt.title("Grayscale Image")
        plt.axis('off')
        plt.show()

        # 2.2. Thresholding (Otsu) + 2.3. Inversion
        import cv2
        import numpy as np
        gray_np = np.array(gray)
        # Otsu's thresholding
        _, thresh = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Inversion: MNIST has dark background, light digit
        inverted = cv2.bitwise_not(thresh)

        print(f"\nThresholded & Inverted Image")
        print(f"Image Width and Height: {inverted.shape[1]} x {inverted.shape[0]}")
        print(f"Number of channels: 1")

        # Display thresholded and inverted image
        plt.figure(figsize=(4, 4))
        plt.imshow(inverted, cmap='gray')
        plt.title("Thresholded & Inverted Image")
        plt.axis('off')
        plt.show()
