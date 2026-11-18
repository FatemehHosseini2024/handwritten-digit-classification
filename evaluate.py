import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            flattened = test_images.view(test_images.size(0), -1)
            output = model(flattened)
            _, predicted = torch.max(output, 1)
            correct += (predicted == test_labels).sum().item()
            total += test_labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(test_labels.cpu().numpy())

    accuracy = correct / total * 100
    return accuracy, all_preds, all_labels


def print_metrics(all_labels, all_preds):
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    print("Per-class metrics:")
    for i in range(10):
        print(f"  Class {i}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    weighted_p, weighted_r, weighted_f, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"\nMacro Average: Precision={macro_p:.4f}, Recall={macro_r:.4f}, F1={macro_f:.4f}")
    print(f"Weighted Average: Precision={weighted_p:.4f}, Recall={weighted_r:.4f}, F1={weighted_f:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    return cm
