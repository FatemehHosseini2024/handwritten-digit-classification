import torch

from model import MNISTNet


def predict(tensor, model_path="best_mnist_model.pth"):
    model = MNISTNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        logits = model(tensor)

    predicted_digit = logits.argmax(dim=1).item()
    print(f"Logits: {[round(v, 2) for v in logits.squeeze().tolist()]}")
    print(f"Predicted Digit: {predicted_digit}")
    return predicted_digit
