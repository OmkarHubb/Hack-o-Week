"""
train.py - Training Pipeline for the Antigravity Levitation Controller
Trains two neural network models:
1. Standard Model: Trained with Mean Squared Error (MSE) Loss + Adam optimizer.
2. Crash-Resilient Model: Trained with Custom Crash-Penalty Loss + Adam optimizer (addresses User Challenge).
Saves models using both Joblib and Pickle, and outputs performance plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import save_and_split_dataset
from neural_network import AntigravityNeuralNetwork, MSELoss


def train_controllers():
    os.makedirs("models", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    dataset_path = os.path.join("data", "antigravity_dataset.npz")
    if not os.path.exists(dataset_path):
        data = save_and_split_dataset()
    else:
        print(f"Loading cached dataset from {dataset_path}...")
        npz = np.load(dataset_path)
        data = {
            "train_X": npz["train_X"],
            "train_y": npz["train_y"],
            "val_X": npz["val_X"],
            "val_y": npz["val_y"],
            "test_X": npz["test_X"],
            "test_y": npz["test_y"],
        }

    X_train, y_train = data["train_X"], data["train_y"]
    X_val, y_val = data["val_X"], data["val_y"]
    X_test, y_test = data["test_X"], data["test_y"]

    print("=" * 60)
    print("PHASE 2 & 3: TRAINING THE NEURAL NETWORK CONTROLLERS")
    print("=" * 60)
    print(f"Training Samples: {len(X_train)} | Val Samples: {len(X_val)} | Test Samples: {len(X_test)}")
    print("Architecture: Input(2) -> Dense(16, ReLU) -> Dense(16, ReLU) -> Output(1, Linear)")

    # -------------------------------------------------------------
    # 1. Train Model A: Standard Controller with MSE Loss
    # -------------------------------------------------------------
    print("\n[1/2] Training Standard Controller (Loss: Mean Squared Error, Optimizer: Adam)...")
    model_mse = AntigravityNeuralNetwork(hidden_sizes=(16, 16), activation="relu", seed=42)
    history_mse = model_mse.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        optimizer_type="adam",
        learning_rate=0.005,
        batch_size=64,
        epochs=120,
        loss_type="mse",
        verbose=True,
    )

    # -------------------------------------------------------------
    # 2. Train Model B: Crash-Resilient Controller (Challenge Loss)
    # -------------------------------------------------------------
    print("\n[2/2] Training Crash-Penalty Controller (User Challenge: High Penalty for Crash Zone)...")
    model_crash = AntigravityNeuralNetwork(hidden_sizes=(16, 16), activation="relu", seed=42)
    history_crash = model_crash.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        optimizer_type="adam",
        learning_rate=0.005,
        batch_size=64,
        epochs=120,
        loss_type="crash_penalty",
        crash_penalty_params={"lambda_crash": 10.0, "y_crash_zone": 0.20, "gamma_underpower": 4.0},
        verbose=True,
    )

    # -------------------------------------------------------------
    # Evaluate on Test Set
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)

    pred_mse = model_mse.predict(X_test)
    pred_crash = model_crash.predict(X_test)

    test_mse_standard = float(np.mean((pred_mse - y_test) ** 2))
    test_mae_standard = float(np.mean(np.abs(pred_mse - y_test)))
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_standard = float(1.0 - (np.sum((pred_mse - y_test) ** 2) / ss_tot))

    test_mse_crash = float(np.mean((pred_crash - y_test) ** 2))
    test_mae_crash = float(np.mean(np.abs(pred_crash - y_test)))
    r2_crash = float(1.0 - (np.sum((pred_crash - y_test) ** 2) / ss_tot))

    # Evaluate specifically in the dangerous low-altitude crash zone (y <= 0.15m)
    crash_zone_mask = X_test[:, 0] <= 0.15
    y_test_cz = y_test[crash_zone_mask]
    pred_mse_cz = pred_mse[crash_zone_mask]
    pred_crash_cz = pred_crash[crash_zone_mask]

    # Under-prediction in crash zone (dangerous under-powering)
    underpower_mse = float(np.mean(np.maximum(0.0, y_test_cz - pred_mse_cz)))
    underpower_crash = float(np.mean(np.maximum(0.0, y_test_cz - pred_crash_cz)))

    print(f"Standard MSE Model  -> Test MSE: {test_mse_standard:.5f} | MAE: {test_mae_standard:.4f} A | R^2: {r2_standard * 100:.2f}%")
    print(f"Crash-Penalty Model -> Test MSE: {test_mse_crash:.5f} | MAE: {test_mae_crash:.4f} A | R^2: {r2_crash * 100:.2f}%")
    print("-" * 60)
    print(f"Crash-Zone Under-Powering Risk (y <= 0.15m):")
    print(f"  Standard MSE Model:  {underpower_mse:.4f} A average deficit")
    print(f"  Crash-Penalty Model: {underpower_crash:.4f} A average deficit (Reduced risk!)")

    # -------------------------------------------------------------
    # Save Models (Phase 4: Joblib and Pickle)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 4: SAVING TRAINED MODELS")
    print("=" * 60)

    model_mse.save("models/antigravity_model_mse.joblib", use_joblib=True)
    model_crash.save("models/antigravity_model_crash_penalty.joblib", use_joblib=True)
    # Primary deployment model
    model_crash.save("models/antigravity_model.joblib", use_joblib=True)
    model_crash.save("models/antigravity_model.pkl", use_joblib=False)

    # -------------------------------------------------------------
    # Plot Training Curves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history_mse["train_loss"]) + 1)

    # Subplot 1: Standard MSE Training & Validation Loss
    axes[0].plot(epochs_range, history_mse["train_loss"], label="Train Loss (MSE)", color="#2563eb", lw=2)
    axes[0].plot(epochs_range, history_mse["val_loss"], label="Validation MSE", color="#f97316", lw=2, linestyle="--")
    axes[0].set_title("Standard Controller Training (MSE Loss)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Mean Squared Error (A²)", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10)

    # Subplot 2: Comparison of Validation MSE over Epochs
    axes[1].plot(epochs_range, history_mse["val_loss"], label="Standard MSE Model Val Loss", color="#2563eb", lw=2)
    axes[1].plot(epochs_range, history_crash["val_loss"], label="Crash-Penalty Model Val Loss", color="#10b981", lw=2)
    axes[1].set_title("Model Convergence Comparison", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Validation MSE (A²)", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    chart_path = os.path.join("assets", "training_loss_curves.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\n[Chart Saved] Training curves saved to: {chart_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_controllers()
