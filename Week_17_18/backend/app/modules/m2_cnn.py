import math
import numpy as np
from typing import Dict, Any, List, Optional

# Pre-generate synthetic yet structurally accurate 28x28 MNIST-like digits (0-9)
# for instant, offline-capable CPU training without requiring external gigabyte downloads.
def generate_synthetic_digits_dataset(n_samples: int = 600, img_size: int = 28):
    """
    Generates realistic 28x28 digit patterns (0-9) with randomized strokes, thickness, and noise.
    """
    np.random.seed(42)
    images = []
    labels = []

    for i in range(n_samples):
        digit = i % 10
        canvas = np.zeros((img_size, img_size), dtype=np.float32)

        # Draw structural stroke features for each digit
        cx, cy = 14, 14
        r = 8 + np.random.uniform(-1, 1)

        if digit == 0:
            # Ellipse / circle
            for theta in np.linspace(0, 2 * np.pi, 60):
                x = int(cx + r * 0.7 * np.cos(theta) + np.random.normal(0, 0.5))
                y = int(cy + r * np.sin(theta) + np.random.normal(0, 0.5))
                if 2 <= x < 26 and 2 <= y < 26:
                    canvas[y-1:y+2, x-1:x+2] = 1.0
        elif digit == 1:
            # Vertical line with slight tilt
            tilt = np.random.uniform(-2, 2)
            for y in range(5, 23):
                x = int(cx + (y - 14) * (tilt / 14.0))
                canvas[y, x-1:x+2] = 1.0
        elif digit == 2:
            # Top arc, diagonal down, bottom horizontal
            for theta in np.linspace(0, np.pi, 30):
                x = int(cx + 6 * np.cos(theta))
                y = int(9 - 4 * np.sin(theta))
                if 2 <= x < 26 and 2 <= y < 26:
                    canvas[y, x-1:x+2] = 1.0
            for t in np.linspace(0, 1, 20):
                x = int(20 - 13 * t)
                y = int(10 + 12 * t)
                canvas[y, x-1:x+2] = 1.0
            canvas[22:24, 7:22] = 1.0
        elif digit == 3:
            # Two right arcs
            for theta in np.linspace(-np.pi/2, np.pi/2, 25):
                x = int(cx + 6 * np.cos(theta))
                y = int(9 + 4 * np.sin(theta))
                canvas[y, x-1:x+2] = 1.0
            for theta in np.linspace(-np.pi/2, np.pi/2, 25):
                x = int(cx + 6 * np.cos(theta))
                y = int(18 + 5 * np.sin(theta))
                canvas[y, x-1:x+2] = 1.0
            canvas[13:15, 12:18] = 1.0
        elif digit == 4:
            # L shape + vertical line
            canvas[5:16, 8:10] = 1.0
            canvas[14:16, 8:21] = 1.0
            canvas[7:23, 18:20] = 1.0
        elif digit == 5:
            # Top bar, left down, bottom arc
            canvas[6:8, 8:20] = 1.0
            canvas[8:14, 8:10] = 1.0
            for theta in np.linspace(-np.pi/2, np.pi/2, 25):
                x = int(cx + 6 * np.cos(theta))
                y = int(17 + 5 * np.sin(theta))
                canvas[y, x-1:x+2] = 1.0
            canvas[13:15, 8:16] = 1.0
        elif digit == 6:
            # Curved down into loop
            for theta in np.linspace(0, 2*np.pi, 40):
                x = int(cx + 6 * np.cos(theta))
                y = int(17 + 5 * np.sin(theta))
                canvas[y, x-1:x+2] = 1.0
            for y in range(6, 17):
                canvas[y, int(8 + (17-y)*0.3):int(10 + (17-y)*0.3)] = 1.0
        elif digit == 7:
            # Top horizontal bar, diagonal down-left
            canvas[6:8, 7:22] = 1.0
            for t in np.linspace(0, 1, 20):
                x = int(21 - 10 * t)
                y = int(8 + 15 * t)
                canvas[y, x-1:x+2] = 1.0
        elif digit == 8:
            # Double loop
            for theta in np.linspace(0, 2*np.pi, 30):
                x = int(cx + 5 * np.cos(theta))
                y = int(9 + 4 * np.sin(theta))
                canvas[y, x-1:x+2] = 1.0
            for theta in np.linspace(0, 2*np.pi, 30):
                x = int(cx + 6 * np.cos(theta))
                y = int(18 + 5 * np.sin(theta))
                canvas[y, x-1:x+2] = 1.0
        else: # 9
            for theta in np.linspace(0, 2*np.pi, 40):
                x = int(cx + 6 * np.cos(theta))
                y = int(10 + 5 * np.sin(theta))
                canvas[y, x-1:x+2] = 1.0
            canvas[10:23, 19:21] = 1.0

        # Add Gaussian noise and blur for realistic variation
        noise = np.random.normal(0, 0.05, canvas.shape)
        canvas = np.clip(canvas + noise, 0.0, 1.0)

        images.append(canvas)
        labels.append(digit)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64)


def apply_conv_filter_demo(filter_type: str, custom_kernel: Optional[List[List[float]]] = None, input_pattern: str = "digit_seven") -> Dict[str, Any]:
    """
    Demonstrates interactive 2D convolution, ReLU, and Max Pooling step-by-step.
    """
    import torch
    import torch.nn.functional as F

    # 1. Create 28x28 input pattern
    img = np.zeros((28, 28), dtype=np.float32)
    if input_pattern == "digit_seven":
        img[5:8, 6:22] = 1.0
        for t in np.linspace(0, 1, 20):
            x = int(21 - 11 * t)
            y = int(8 + 15 * t)
            img[y-1:y+2, x-1:x+2] = 1.0
    elif input_pattern == "cross":
        img[12:16, 4:24] = 1.0
        img[4:24, 12:16] = 1.0
    elif input_pattern == "checkerboard":
        for r in range(4):
            for c in range(4):
                if (r + c) % 2 == 0:
                    img[r*7:(r+1)*7, c*7:(c+1)*7] = 1.0
    else: # diagonal
        for d in range(28):
            img[d, max(0, d-2):min(28, d+3)] = 1.0

    # 2. Select 3x3 Filter Kernel
    if filter_type == "sobel_x":
        kernel = [[-1.0, 0.0, 1.0],
                  [-2.0, 0.0, 2.0],
                  [-1.0, 0.0, 1.0]]
        explanation = "Sobel Vertical Edge Detector: Measures horizontal intensity gradient dI/dx, highlighting vertical borders."
    elif filter_type == "sobel_y":
        kernel = [[-1.0, -2.0, -1.0],
                  [ 0.0,  0.0,  0.0],
                  [ 1.0,  2.0,  1.0]]
        explanation = "Sobel Horizontal Edge Detector: Measures vertical intensity gradient dI/dy, highlighting horizontal lines."
    elif filter_type == "sharpen":
        kernel = [[ 0.0, -1.0,  0.0],
                  [-1.0,  5.0, -1.0],
                  [ 0.0, -1.0,  0.0]]
        explanation = "Sharpen Filter: Amplifies center pixel relative to immediate 4-connected neighbors, boosting local contrast."
    elif filter_type == "gaussian_blur":
        kernel = [[1/16, 2/16, 1/16],
                  [2/16, 4/16, 2/16],
                  [1/16, 2/16, 1/16]]
        explanation = "Gaussian Blur / Smoothing: Averages surrounding neighborhood to suppress high-frequency noise."
    elif filter_type == "ridge":
        kernel = [[-1.0, -1.0, -1.0],
                  [-1.0,  8.0, -1.0],
                  [-1.0, -1.0, -1.0]]
        explanation = "Laplacian / Ridge Detector: Second derivative filter highlighting boundaries in all directions."
    elif custom_kernel and len(custom_kernel) == 3 and len(custom_kernel[0]) == 3:
        kernel = custom_kernel
        explanation = "Custom User Kernel: Real-time user specified 3x3 convolution matrix."
    else:
        kernel = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
        explanation = "Standard Edge Kernel."

    # 3. Apply PyTorch operations
    t_input = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 28, 28)
    t_kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 3, 3)

    # Conv2d with padding=1 keeps dimension 28x28
    t_conv = F.conv2d(t_input, t_kernel, padding=1)
    t_relu = F.relu(t_conv)
    t_pool = F.max_pool2d(t_relu, kernel_size=2, stride=2) # 14x14

    return {
        "filter_name": filter_type,
        "explanation": explanation,
        "kernel": kernel,
        "input_pattern": input_pattern,
        "input_shape": list(t_input.shape),
        "conv_shape": list(t_conv.shape),
        "relu_shape": list(t_relu.shape),
        "pool_shape": list(t_pool.shape),
        "input_matrix": [[round(float(v), 2) for v in row] for row in img.tolist()],
        "conv_matrix": [[round(float(v), 2) for v in row] for row in t_conv.squeeze().tolist()],
        "relu_matrix": [[round(float(v), 2) for v in row] for row in t_relu.squeeze().tolist()],
        "pool_matrix": [[round(float(v), 2) for v in row] for row in t_pool.squeeze().tolist()],
        "summary": {
            "conv_min": round(float(t_conv.min()), 3),
            "conv_max": round(float(t_conv.max()), 3),
            "relu_zeros_pct": round(float((t_relu == 0).float().mean() * 100), 1),
            "pool_downsample_ratio": "4x reduction (28x28=784 -> 14x14=196 values)"
        }
    }


# Singleton cache to store trained CNN model and test dataset in memory
_CACHED_CNN_MODEL = None
_CACHED_TEST_DATA = None

def train_cnn_classifier(dataset_name: str = "mnist", epochs: int = 5, learning_rate: float = 0.002, batch_size: int = 32) -> Dict[str, Any]:
    """
    Trains a real PyTorch CNN (Conv2D -> BatchNorm -> ReLU -> MaxPool -> Conv2D -> Dense)
    on 28x28 digits. CPU-friendly, fast convergence in ~2-3 seconds.
    """
    global _CACHED_CNN_MODEL, _CACHED_TEST_DATA
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(42)

    # 1. Dataset prep
    X, y = generate_synthetic_digits_dataset(n_samples=500)
    split = 400
    X_train, y_train = torch.tensor(X[:split]).unsqueeze(1), torch.tensor(y[:split])
    X_val, y_val = torch.tensor(X[split:]).unsqueeze(1), torch.tensor(y[split:])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # 2. Define CNN Architecture
    class DigitCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2) # 14x14

            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2) # 7x7

            self.fc1 = nn.Linear(32 * 7 * 7, 64)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            f1 = self.pool1(self.relu1(self.bn1(self.conv1(x))))
            f2 = self.pool2(self.relu2(self.bn2(self.conv2(f1))))
            flat = f2.view(f2.size(0), -1)
            h = self.relu3(self.fc1(flat))
            out = self.fc2(h)
            return out, f1, f2

    model = DigitCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits, _, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits, _, _ = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
            val_preds = torch.argmax(val_logits, dim=1)
            val_acc = (val_preds == y_val).float().mean().item() * 100.0

        history.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 2),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 2)
        })

    # Cache model and validation data for instant interactive inference
    _CACHED_CNN_MODEL = model
    _CACHED_TEST_DATA = (X_val, y_val)

    # Generate 6 sample predictions with probabilities
    sample_preds = []
    with torch.no_grad():
        test_logits, _, _ = model(X_val[:6])
        probs = torch.softmax(test_logits, dim=1).tolist()
        for idx in range(6):
            pred_class = int(np.argmax(probs[idx]))
            true_class = int(y_val[idx].item())
            sample_preds.append({
                "sample_id": idx,
                "true_label": true_class,
                "predicted_label": pred_class,
                "confidence": round(probs[idx][pred_class] * 100.0, 1),
                "is_correct": (pred_class == true_class),
                "top_probabilities": [{"digit": d, "prob": round(probs[idx][d] * 100.0, 1)} for d in range(10)]
            })

    return {
        "status": "success",
        "dataset": dataset_name.upper(),
        "total_train_samples": len(X_train),
        "total_val_samples": len(X_val),
        "epochs": epochs,
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "history": history,
        "sample_predictions": sample_preds,
        "architecture_summary": [
            {"layer": "Input", "shape": "1 x 28 x 28", "params": 0},
            {"layer": "Conv2d (1->16, k=3, p=1)", "shape": "16 x 28 x 28", "params": 160},
            {"layer": "BatchNorm2d + ReLU + MaxPool(2x2)", "shape": "16 x 14 x 14", "params": 32},
            {"layer": "Conv2d (16->32, k=3, p=1)", "shape": "32 x 14 x 14", "params": 4640},
            {"layer": "BatchNorm2d + ReLU + MaxPool(2x2)", "shape": "32 x 7 x 7", "params": 64},
            {"layer": "Dense FC1 (1568 -> 64) + ReLU", "shape": "64", "params": 100416},
            {"layer": "Dense FC2 (64 -> 10) Logits", "shape": "10", "params": 650}
        ]
    }


def predict_cnn_sample(sample_index: int = 0) -> Dict[str, Any]:
    """
    Performs inference on a specific sample and extracts intermediate Conv1 and Conv2 feature maps.
    """
    global _CACHED_CNN_MODEL, _CACHED_TEST_DATA
    if _CACHED_CNN_MODEL is None or _CACHED_TEST_DATA is None:
        # Pre-train quick model if not yet cached
        train_cnn_classifier(epochs=3)

    import torch

    X_val, y_val = _CACHED_TEST_DATA
    idx = max(0, min(sample_index, len(X_val) - 1))
    sample_tensor = X_val[idx:idx+1]
    true_label = int(y_val[idx].item())

    _CACHED_CNN_MODEL.eval()
    with torch.no_grad():
        logits, f1, f2 = _CACHED_CNN_MODEL(sample_tensor)
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_label = int(np.argmax(probs))

    # Extract 4 sample feature map slices from Conv1 (14x14) and Conv2 (7x7)
    f1_channels = []
    for c in range(min(4, f1.shape[1])):
        f1_channels.append({
            "channel_id": c,
            "matrix": [[round(float(v), 2) for v in row] for row in f1[0, c].tolist()]
        })

    f2_channels = []
    for c in range(min(4, f2.shape[1])):
        f2_channels.append({
            "channel_id": c,
            "matrix": [[round(float(v), 2) for v in row] for row in f2[0, c].tolist()]
        })

    input_img = sample_tensor.squeeze().tolist()

    return {
        "sample_index": idx,
        "true_label": true_label,
        "predicted_label": pred_label,
        "confidence": round(probs[pred_label] * 100.0, 1),
        "probabilities": [{"digit": d, "prob": round(probs[d] * 100.0, 1)} for d in range(10)],
        "input_image": [[round(float(v), 2) for v in row] for row in input_img],
        "conv1_feature_maps": f1_channels,
        "conv2_feature_maps": f2_channels
    }
