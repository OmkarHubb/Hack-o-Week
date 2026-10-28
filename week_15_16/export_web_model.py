"""
export_web_model.py - Exports trained NumPy neural network weights to JSON
Enables the interactive HTML5/Canvas Antigravity Chamber web simulator to run
the trained model with 60 FPS real-time client-side inference.
"""

import os
import json
import numpy as np
from neural_network import AntigravityNeuralNetwork
from physics import MagneticLevitationChamber


def export_model_to_json(
    model_path: str = "models/antigravity_model.joblib",
    output_path: str = "web/model_weights.json",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model = AntigravityNeuralNetwork.load(model_path)
    chamber = MagneticLevitationChamber()

    layers_data = []
    for layer in model.layers:
        layers_data.append({
            "in_features": layer.in_features,
            "out_features": layer.out_features,
            "activation": layer.activation_name,
            "weights": layer.W.tolist(),
            "biases": layer.b.tolist()[0],
        })

    payload = {
        "architecture": {
            "hidden_sizes": model.hidden_sizes,
            "activation": model.activation,
        },
        "scaler": {
            "mean": model.scaler.mean.tolist()[0],
            "std": model.scaler.std.tolist()[0],
        },
        "layers": layers_data,
        "physics": {
            "H": chamber.H,
            "y_target": chamber.y_target,
            "m": chamber.m,
            "g": chamber.g,
            "k_mag": chamber.k_mag,
            "c_drag": chamber.c_drag,
            "eps": chamber.eps,
            "max_current": chamber.max_current,
            "equilibrium_current": chamber.equilibrium_current,
            "dt": chamber.dt,
        },
    }

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[Exported] Neural network weights & physics config exported to: {output_path}")


if __name__ == "__main__":
    export_model_to_json()
