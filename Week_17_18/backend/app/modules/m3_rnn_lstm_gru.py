import time
import numpy as np
from typing import Dict, Any, List

def generate_time_series_data(seq_len: int = 20, total_points: int = 300):
    """
    Generates a realistic multi-frequency non-linear time series signal.
    """
    t = np.linspace(0, 30, total_points)
    signal = np.sin(0.8 * t) + 0.5 * np.cos(1.7 * t) + 0.1 * np.random.normal(0, 0.2, total_points)

    X_seqs = []
    y_targets = []
    for i in range(len(signal) - seq_len):
        X_seqs.append(signal[i:i+seq_len])
        y_targets.append(signal[i+seq_len])

    return np.array(X_seqs, dtype=np.float32), np.array(y_targets, dtype=np.float32), signal


def compare_recurrent_architectures(epochs: int = 20, hidden_size: int = 32, sequence_length: int = 20) -> Dict[str, Any]:
    """
    Trains real PyTorch Vanilla RNN, LSTM, and GRU side-by-side on time-series forecasting.
    Measures speed, convergence, parameter count, and prediction accuracy.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    np.random.seed(42)

    X_data, y_data, raw_signal = generate_time_series_data(seq_len=sequence_length, total_points=260)
    split = 200

    # Reshape to (Batch, Seq_len, Input_size=1)
    X_train = torch.tensor(X_data[:split]).unsqueeze(-1)
    y_train = torch.tensor(y_data[:split]).unsqueeze(-1)
    X_test = torch.tensor(X_data[split:]).unsqueeze(-1)
    y_test = torch.tensor(y_data[split:]).unsqueeze(-1)

    # 1. Model Definitions
    class SeqModel(nn.Module):
        def __init__(self, rnn_type: str):
            super().__init__()
            self.rnn_type = rnn_type
            if rnn_type == "rnn":
                self.cell = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
            elif rnn_type == "lstm":
                self.cell = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
            elif rnn_type == "gru":
                self.cell = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.cell(x)
            last_hidden = out[:, -1, :] # Take final time step hidden representation
            return self.head(last_hidden)

    models = {
        "RNN": SeqModel("rnn"),
        "LSTM": SeqModel("lstm"),
        "GRU": SeqModel("gru")
    }

    criterion = nn.MSELoss()
    comparison_results = {}

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        history = []
        param_count = sum(p.numel() for p in model.parameters())

        start_time = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()

            # Eval on test set
            model.eval()
            with torch.no_grad():
                test_preds = model(X_test)
                val_loss = criterion(test_preds, y_test).item()

            history.append({
                "epoch": ep,
                "train_loss": round(float(loss.item()), 5),
                "val_loss": round(float(val_loss), 5)
            })

        duration_ms = round((time.time() - start_time) * 1000, 1)

        # Get test predictions for plotting
        model.eval()
        with torch.no_grad():
            final_test_preds = model(X_test).squeeze().tolist()

        comparison_results[name] = {
            "param_count": param_count,
            "train_duration_ms": duration_ms,
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1]["val_loss"],
            "history": history,
            "test_predictions": [round(float(v), 3) for v in final_test_preds[:30]]
        }

    # Ground truth values for the test slice
    ground_truth = [round(float(v), 3) for v in y_test.squeeze().tolist()[:30]]

    return {
        "sequence_length": sequence_length,
        "hidden_size": hidden_size,
        "epochs": epochs,
        "ground_truth": ground_truth,
        "models": comparison_results,
        "gating_mechanisms_summary": {
            "RNN": {
                "gates": 0,
                "hidden_state": "h_t = tanh(W_h * h_{t-1} + W_x * x_t + b)",
                "strengths": "Fastest computation, minimal parameters.",
                "weaknesses": "Suffers from vanishing gradients over sequences > 10-15 steps."
            },
            "LSTM": {
                "gates": 3,
                "gates_list": ["Forget Gate (f_t)", "Input Gate (i_t)", "Output Gate (o_t)"],
                "cell_state": "C_t = f_t * C_{t-1} + i_t * C~_t",
                "strengths": "Additive error carousel prevents gradient decay; excels at long-term temporal memory.",
                "weaknesses": "Heaviest compute, 4x parameters compared to standard RNN."
            },
            "GRU": {
                "gates": 2,
                "gates_list": ["Reset Gate (r_t)", "Update Gate (z_t)"],
                "hidden_state": "h_t = (1 - z_t) * h_{t-1} + z_t * h~_t",
                "strengths": "Near-LSTM memory quality with ~25% fewer parameters and faster training.",
                "weaknesses": "Lacks separate cell memory highway, may slightly underperform LSTM on very long complex sequences."
            }
        }
    }
