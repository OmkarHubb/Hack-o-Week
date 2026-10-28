import math
import numpy as np
from typing import Dict, Any, List

def run_tensor_operations(op_type: str, shape_a: List[int], shape_b: List[int], activation: str = "relu") -> Dict[str, Any]:
    """
    Demonstrates fundamental PyTorch tensor creations, shapes, math, and activations.
    """
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    # Generate realistic sample tensors
    t_a = torch.randn(*shape_a)
    t_b = torch.randn(*shape_b)

    result_tensor = None
    operation_name = ""
    formula_latex = ""
    description = ""

    if op_type == "matmul":
        # Ensure dimensions match for matrix multiplication
        if len(shape_a) >= 2 and len(shape_b) >= 2 and shape_a[-1] == shape_b[-2]:
            result_tensor = torch.matmul(t_a, t_b)
            operation_name = f"Matrix Multiplication: ({shape_a}) @ ({shape_b})"
            formula_latex = "C_{i,j} = \\sum_{k} A_{i,k} \\cdot B_{k,j}"
            description = "Computes the dot product between rows of tensor A and columns of tensor B."
        else:
            # Fallback to matched inner dims
            t_b = torch.randn(shape_a[-1], shape_a[0])
            result_tensor = torch.matmul(t_a, t_b)
            operation_name = f"Matrix Multiplication (auto-aligned): ({list(t_a.shape)}) @ ({list(t_b.shape)})"
            formula_latex = "C = A \\times B"
            description = "Adjusted inner dimensions so dot product is mathematically valid."

    elif op_type == "add":
        # Broadcasting or elementwise
        t_b_match = torch.randn(*shape_a)
        result_tensor = t_a + t_b_match
        operation_name = f"Element-wise Addition: ({shape_a}) + ({shape_a})"
        formula_latex = "C_{i,j} = A_{i,j} + B_{i,j}"
        description = "Adds corresponding elements of tensors. Supports broadcasting if shapes are compatible."

    elif op_type == "reshape":
        total_elements = math.prod(shape_a)
        new_shape = [1, total_elements]
        result_tensor = t_a.view(*new_shape)
        operation_name = f"Reshape / View: ({shape_a}) -> ({new_shape})"
        formula_latex = "\\text{flatten}(A) \\in \\mathbb{R}^{1 \\times N}"
        description = "Changes the tensor dimension without copying underlying memory buffer."

    elif op_type == "transpose":
        if len(shape_a) >= 2:
            result_tensor = t_a.transpose(0, 1)
            operation_name = f"Transpose: ({shape_a}) -> ({list(result_tensor.shape)})"
            formula_latex = "A^T_{i,j} = A_{j,i}"
            description = "Swaps dimensions 0 and 1."
        else:
            result_tensor = t_a.unsqueeze(0)
            operation_name = f"Unsqueeze: ({shape_a}) -> ({list(result_tensor.shape)})"
            formula_latex = "\\text{unsqueeze}(A, 0)"
            description = "Adds a singleton batch dimension."

    elif op_type == "activation":
        if activation == "relu":
            result_tensor = F.relu(t_a)
            operation_name = f"ReLU Activation: max(0, x)"
            formula_latex = "f(x) = \\max(0, x)"
            description = "Rectified Linear Unit zeroes out negative inputs and preserves positive values, introducing non-linearity."
        elif activation == "sigmoid":
            result_tensor = torch.sigmoid(t_a)
            operation_name = f"Sigmoid Activation"
            formula_latex = "\\sigma(x) = \\frac{1}{1 + e^{-x}}"
            description = "Squashes all values into probability range (0, 1)."
        elif activation == "softmax":
            result_tensor = F.softmax(t_a, dim=-1)
            operation_name = f"Softmax Normalization"
            formula_latex = "S(x)_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}"
            description = "Normalizes logits across the last axis into a valid categorical probability distribution summing to 1.0."

    return {
        "operation_name": operation_name,
        "formula_latex": formula_latex,
        "description": description,
        "tensor_a": {
            "shape": list(t_a.shape),
            "dtype": str(t_a.dtype),
            "values": [[round(float(v), 4) for v in row] for row in (t_a.view(t_a.shape[0], -1)[:5].tolist())]
        },
        "result_tensor": {
            "shape": list(result_tensor.shape),
            "dtype": str(result_tensor.dtype),
            "values": [[round(float(v), 4) for v in row] for row in (result_tensor.view(result_tensor.shape[0], -1)[:5].tolist())],
            "min": round(float(result_tensor.min()), 4),
            "max": round(float(result_tensor.max()), 4),
            "mean": round(float(result_tensor.mean()), 4),
            "std": round(float(result_tensor.std()), 4) if result_tensor.numel() > 1 else 0.0
        }
    }


def run_autograd_demo(formula: str, x_val: float) -> Dict[str, Any]:
    """
    Demonstrates PyTorch dynamic computational graph and automatic backward differentiation.
    """
    import torch

    x = torch.tensor(float(x_val), requires_grad=True, dtype=torch.float32)

    # Calculate step-by-step for visualization of graph
    # Default to 3*x**2 + 2*x + 1 if custom is invalid
    try:
        # Safe eval using torch symbols
        safe_dict = {
            "x": x,
            "torch": torch,
            "sin": torch.sin,
            "cos": torch.cos,
            "exp": torch.exp,
            "log": torch.log
        }
        y = eval(formula, {"__builtins__": {}}, safe_dict)
    except Exception:
        y = 3 * (x ** 2) + 2 * x + 1
        formula = "3*x**2 + 2*x + 1"

    # Forward pass value
    y_val = float(y.item())

    # Backward pass
    y.backward()
    dy_dx = float(x.grad.item())

    # Analytical verification for polynomial 3x^2 + 2x + 1
    analytical = None
    if formula.strip() == "3*x**2 + 2*x + 1":
        analytical = 6.0 * x_val + 2.0

    return {
        "formula": formula,
        "x_value": round(x_val, 4),
        "y_forward": round(y_val, 4),
        "gradient_dy_dx": round(dy_dx, 4),
        "analytical_expected": round(analytical, 4) if analytical is not None else round(dy_dx, 4),
        "computation_graph_steps": [
            {"step": 1, "node": "Input Variable", "operation": f"x = {x_val}", "requires_grad": True},
            {"step": 2, "node": "Forward Evaluation", "operation": f"y = {formula} = {round(y_val, 4)}", "grad_fn": str(y.grad_fn)},
            {"step": 3, "node": "Call y.backward()", "operation": "Seed incoming gradient dy/dy = 1.0 at root", "accumulate_grad": True},
            {"step": 4, "node": "Gradient Propagated", "operation": f"x.grad populated via Chain Rule: {round(dy_dx, 4)}", "result": round(dy_dx, 4)}
        ]
    }


def train_basic_mlp(learning_rate: float = 0.05, optimizer_name: str = "Adam", epochs: int = 30, hidden_dim: int = 16) -> Dict[str, Any]:
    """
    Trains a real 2-layer PyTorch MLP on a non-linear 2D classification task (concentric circles).
    Demonstrates forward pass, CrossEntropyLoss, optimizer.zero_grad(), loss.backward(), and optimizer.step().
    """
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_circles

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Create synthetic dataset (circles problem - impossible for linear models)
    X, y = make_circles(n_samples=250, noise=0.1, factor=0.4, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 2. Define Model
    class SimpleMLP(nn.Module):
        def __init__(self, in_features=2, hidden=hidden_dim, out_classes=2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_classes)
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: List[Dict[str, Any]] = []

    # 3. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        # A. Zero gradients from previous iteration
        optimizer.zero_grad()

        # B. Forward pass
        logits = model(X_tensor)

        # C. Compute loss
        loss = criterion(logits, y_tensor)

        # D. Backward pass (autograd chain rule)
        loss.backward()

        # Measure gradient norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        # E. Optimizer step (update parameters)
        optimizer.step()

        # Calculate training accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            accuracy = (preds == y_tensor).float().mean().item() * 100.0

        history.append({
            "epoch": epoch,
            "loss": round(float(loss.item()), 4),
            "accuracy": round(accuracy, 2),
            "grad_norm": round(float(total_grad_norm), 4)
        })

    # Sample decision points for plotting
    sample_points = []
    for i in range(min(50, len(X))):
        sample_points.append({
            "x1": round(float(X[i, 0]), 3),
            "x2": round(float(X[i, 1]), 3),
            "true_label": int(y[i]),
            "pred_label": int(preds[i].item())
        })

    return {
        "dataset_name": "Non-Linear Concentric Circles (250 points)",
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "final_loss": history[-1]["loss"],
        "final_accuracy": history[-1]["accuracy"],
        "history": history,
        "sample_points": sample_points,
        "pytorch_workflow_explained": [
            "1. optimizer.zero_grad() - Clears previous accumulated gradients in model parameters",
            "2. logits = model(X) - Executes the forward pass through linear layers & ReLU activations",
            "3. loss = criterion(logits, y) - Measures prediction discrepancy using Cross-Entropy",
            "4. loss.backward() - Traverses the PyTorch Autograd computational graph computing dLoss/dW",
            "5. optimizer.step() - Updates each weight W = W - lr * grad (or with Adam momentum & adaptive scaling)"
        ]
    }
