import torch
import torch.nn as nn


def train(model, train_loader, val_loader, criterion, optimizer, epochs=15, patience=3):
    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad()
            flattened = batch_images.view(batch_images.size(0), -1)
            output = model(flattened)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_images.size(0)

        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

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

        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.6f} - "
              f"Val Loss: {avg_val_loss:.6f} - "
              f"Val Accuracy: {val_accuracy:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            print("  -> Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model restored (best val loss: {best_val_loss:.6f}).")

    return train_losses, val_losses, val_accuracies, best_val_loss
