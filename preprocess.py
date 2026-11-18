import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File does not exist: {image_path}")
    img = Image.open(image_path)
    img.load()
    print(f"Original Image")
    print(f"Image Width and Height: {img.width} x {img.height}")
    print(f"Image Mode: {img.mode}")
    print(f"Number of channels: {len(img.getbands())}")
    return img


def preprocess(img):
    # 2.1 Grayscale
    gray = img.convert('L')
    print(f"\nGrayscale Image: {gray.width} x {gray.height}, Mode: {gray.mode}")

    # 2.2 Thresholding (Otsu) + 2.3 Inversion
    gray_np = np.array(gray)
    _, thresh = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)

    # 3. Digit Detection & Cropping
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found. Cannot crop digit.")
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = inverted[y:y+h, x:x+w]
    print(f"Contours: {len(contours)}, Bounding box: x={x}, y={y}, w={w}, h={h}")
    print(f"Cropped: {w} x {h}")

    # 4. Resizing & Centering (MNIST format)
    if h > w:
        new_h = 20
        new_w = int(w * 20 / h)
    else:
        new_w = 20
        new_h = int(h * 20 / w)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    y_offset = (28 - new_h) // 2
    x_offset = (28 - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    print(f"Resized: {new_w} x {new_h}, Canvas: 28 x 28, Padding: top={y_offset}, left={x_offset}")

    # 5. Tensor Conversion & Normalization
    tensor = transforms.ToTensor()(Image.fromarray(canvas))
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    tensor = normalize(tensor).unsqueeze(0)
    print(f"Tensor: {list(tensor.shape)}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

    return tensor
