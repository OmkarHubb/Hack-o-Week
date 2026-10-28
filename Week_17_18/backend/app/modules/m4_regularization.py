import copy
import numpy as np
from typing import Dict, Any, List

def run_regularization_experiment(dropout_rate: float = 0.4, epochs: int = 40, learning_rate: float = 0.01, weight_decay: float = 0.0001) -> Dict[str, Any]:
    """
    Demonstrates overfitting vs Dropout vs BatchNorm vs Early Stopping on an overparameterized MLP.
    Returns real training and validation curves, generalization gaps, and early stopping triggers.
    """
    import torch
    import torch.nn as nn
    from sklearn.datasets import make_classification

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Create a dataset prone to overfitting: few samples (120), noisy features (20)
    X_raw, y_raw = make_classification(
        n_samples=160,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        flip_y=0.15, # Label noise to incentivize memorization
        random_state=42
    )

    split = 90
    X_train = torch.tensor(X_raw[:split], dtype=torch.float32)
    y_train = torch.tensor(y_raw[:split], dtype=torch.long)
    X_val = torch.tensor(X_raw[split:], dtype=torch.float32)
    y_val = torch.tensor(y_raw[split:], dtype=torch.long)

    # 2. Model Factory
    def build_model(use_dropout: bool = False, use_bn: bool = False):
        layers = []
        hidden_sizes = [20, 96, 96, 2]

        for i in range(len(hidden_sizes) - 2):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-2], hidden_sizes[-1]))
        return nn.Sequential(*layers)

    configurations = [
        {"name": "Baseline (No Regularization)", "dropout": False, "bn": False, "early_stop": False, "l2": 0.0},
        {"name": "Dropout (p=" + str(dropout_rate) + ")", "dropout": True, "bn": False, "early_stop": False, "l2": 0.0},
        {"name": "Batch Normalization", "dropout": False, "bn": True, "early_stop": False, "l2": 0.0},
        {"name": "Early Stopping (Patience=6)", "dropout": False, "bn": False, "early_stop": True, "l2": 0.0}
    ]

    criterion = nn.CrossEntropyLoss()
    results = {}

    for cfg in configurations:
        model = build_model(use_dropout=cfg["dropout"], use_bn=cfg["bn"])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=cfg["l2"])

        history = []
        best_val_loss = float("inf")
        patience = 6
        patience_counter = 0
        stopped_epoch = None
        best_weights = None

        for ep in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            train_acc = (torch.argmax(logits, dim=1) == y_train).float().mean().item() * 100.0

            # Evaluation mode (disables dropout, fixes batch norm running stats)
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = criterion(val_logits, y_val).item()
                val_acc = (torch.argmax(val_logits, dim=1) == y_val).float().mean().item() * 100.0

            history.append({
                "epoch": ep,
                "train_loss": round(float(loss.item()), 4),
                "val_loss": round(float(val_loss), 4),
                "train_acc": round(float(train_acc), 1),
                "val_acc": round(float(val_acc), 1)
            })

            # Check Early Stopping
            if cfg["early_stop"]:
                if val_loss < best_val_loss - 0.005:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_weights = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience and stopped_epoch is None:
                        stopped_epoch = ep
                        break

        # Calculate final metrics and generalization gap
        final_train_acc = history[-1]["train_acc"]
        final_val_acc = history[-1]["val_acc"]
        gen_gap = round(final_train_acc - final_val_acc, 1)

        results[cfg["name"]] = {
            "stopped_epoch": stopped_epoch if stopped_epoch else epochs,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "final_train_acc": final_train_acc,
            "final_val_acc": final_val_acc,
            "generalization_gap": gen_gap,
            "history": history
        }

    return {
        "dataset_summary": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "features": 20,
            "classes": 2,
            "noise_level": "High (15% random label flips)"
        },
        "configurations": results,
        "theoretical_insights": [
            {
                "technique": "Dropout",
                "formula": "r_j \\sim \\text{Bernoulli}(1-p), \\quad \\tilde{y} = r \\odot y / (1-p)",
                "how_it_works": "Temporarily zeroes random activations during training. Prevents co-adaptation of weights and acts as an implicit ensemble of 2^N sub-networks."
            },
            {
                "technique": "Batch Normalization",
                "formula": "\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}, \\quad y_i = \\gamma \\hat{x}_i + \\beta",
                "how_it_works": "Normalizes activations per mini-batch, stabilizing internal covariate shift and smoothing the loss landscape for faster training."
            },
            {
                "technique": "Early Stopping",
                "formula": "\\text{Halt when } \\mathcal{L}_{val}(t) \\ge \\min_{\\tau \\le t} \\mathcal{L}_{val}(\\tau) \\text{ for } k \\text{ consecutive epochs}",
                "how_it_works": "Monitors validation loss; halts training before the network begins memorizing noise, saving optimal checkpoint weights."
            }
        ]
    }
