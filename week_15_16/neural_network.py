"""
neural_network.py - Multi-Layer Perceptron (MLP) Neural Network from Scratch in NumPy
Features:
- Modular Dense Layers with Xavier/Glorot and He/Kaiming weight initialization.
- Activation Functions: ReLU, LeakyReLU, Sigmoid, Tanh, Linear.
- Optimizers: Stochastic Gradient Descent (SGD) with momentum, and Adam.
- Loss Functions:
    * Standard Mean Squared Error (MSE)
    * Custom Crash-Penalty Loss (Challenge: Penalizes proximity to floor and under-powering near crash boundary)
- Feature normalization (StandardScaler) built into the pipeline.
- Saving/Loading via Pickle and Joblib.
"""

import pickle
import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# =====================================================================
# Activation Functions with Analytical Derivatives
# =====================================================================

class Activation:
    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        return (z > 0.0).astype(np.float64)

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        s = Activation.sigmoid(z)
        return s * (1.0 - s)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        t = np.tanh(z)
        return 1.0 - t ** 2

    @staticmethod
    def linear(z: np.ndarray) -> np.ndarray:
        return z

    @staticmethod
    def linear_derivative(z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)


# =====================================================================
# Dense (Fully-Connected) Layer
# =====================================================================

class DenseLayer:
    """
    Fully-connected linear layer: Z = X @ W + b
    """

    def __init__(self, in_features: int, out_features: int, activation: str = "relu", seed: int = None):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation.lower()

        rng = np.random.RandomState(seed)
        # Weight Initialization: He for ReLU, Xavier/Glorot for Sigmoid/Tanh/Linear
        if self.activation_name in ("relu", "leaky_relu"):
            limit = np.sqrt(2.0 / in_features)
            self.W = rng.randn(in_features, out_features) * limit
        else:
            limit = np.sqrt(2.0 / (in_features + out_features))
            self.W = rng.randn(in_features, out_features) * limit

        self.b = np.zeros((1, out_features), dtype=np.float64)

        # Cache for backpropagation
        self.input_cache = None
        self.z_cache = None

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_cache = x
        self.z_cache = np.dot(x, self.W) + self.b

        if self.activation_name == "relu":
            return Activation.relu(self.z_cache)
        elif self.activation_name == "sigmoid":
            return Activation.sigmoid(self.z_cache)
        elif self.activation_name == "tanh":
            return Activation.tanh(self.z_cache)
        elif self.activation_name == "linear":
            return Activation.linear(self.z_cache)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Backpropagate gradients through activation and linear weights.
        d_out: gradient of loss w.r.t layer output A
        Returns: gradient of loss w.r.t layer input X
        """
        if self.activation_name == "relu":
            d_act = Activation.relu_derivative(self.z_cache)
        elif self.activation_name == "sigmoid":
            d_act = Activation.sigmoid_derivative(self.z_cache)
        elif self.activation_name == "tanh":
            d_act = Activation.tanh_derivative(self.z_cache)
        elif self.activation_name == "linear":
            d_act = Activation.linear_derivative(self.z_cache)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")

        dZ = d_out * d_act  # Shape: (N, out_features)
        N = self.input_cache.shape[0]

        self.dW = np.dot(self.input_cache.T, dZ) / N
        self.db = np.sum(dZ, axis=0, keepdims=True) / N

        dX = np.dot(dZ, self.W.T)  # Shape: (N, in_features)
        return dX


# =====================================================================
# Optimizers: SGD (with Momentum) and Adam
# =====================================================================

class SGDOptimizer:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v_W = []
        self.v_b = []
        self.initialized = False

    def init_state(self, layers: list[DenseLayer]):
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.initialized = True

    def step(self, layers: list[DenseLayer]):
        if not self.initialized:
            self.init_state(layers)

        for i, layer in enumerate(layers):
            self.v_W[i] = self.momentum * self.v_W[i] + (1.0 - self.momentum) * layer.dW
            self.v_b[i] = self.momentum * self.v_b[i] + (1.0 - self.momentum) * layer.db
            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]


class AdamOptimizer:
    def __init__(self, lr: float = 0.005, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m_W = []
        self.v_W = []
        self.m_b = []
        self.v_b = []
        self.initialized = False

    def init_state(self, layers: list[DenseLayer]):
        self.m_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.m_b = [np.zeros_like(layer.b) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.t = 0
        self.initialized = True

    def step(self, layers: list[DenseLayer]):
        if not self.initialized:
            self.init_state(layers)

        self.t += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        for i, layer in enumerate(layers):
            # Update biased 1st and 2nd moment estimates
            self.m_W[i] = self.beta1 * self.m_W[i] + (1.0 - self.beta1) * layer.dW
            self.v_W[i] = self.beta2 * self.v_W[i] + (1.0 - self.beta2) * (layer.dW ** 2)

            self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * layer.db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (layer.db ** 2)

            # Weight update
            layer.W -= lr_t * self.m_W[i] / (np.sqrt(self.v_W[i]) + self.eps)
            layer.b -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.eps)


# =====================================================================
# Loss Functions: MSE and Challenge Custom Crash-Penalty Loss
# =====================================================================

class MSELoss:
    @staticmethod
    def forward(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    @staticmethod
    def backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return 2.0 * (y_pred - y_true)


class CrashPenaltyLoss:
    """
    Challenge for the User:
    Custom loss function that penalizes the model significantly more if the
    object is near the crash boundary (y <= 0) and current is under-predicted,
    while treating slight wobbles near the target height with standard tolerance.

    Loss formula:
    w_i = 1.0 + lambda_crash * exp(-y_phys / y_crash_zone) * (1.0 + gamma * (y_pred < y_true))
    Loss = mean(w_i * (y_pred - y_true)^2)

    Derivative w.r.t y_pred:
    dLoss / dy_pred = 2.0 * w_i * (y_pred - y_true)
    """

    def __init__(self, lambda_crash: float = 8.0, y_crash_zone: float = 0.25, gamma_underpower: float = 3.0):
        self.lambda_crash = lambda_crash
        self.y_crash_zone = y_crash_zone
        self.gamma_underpower = gamma_underpower

    def compute_weights(self, y_pred: np.ndarray, y_true: np.ndarray, distances: np.ndarray) -> np.ndarray:
        """
        distances: physical distance from floor (y in meters), shape (N, 1)
        """
        # Proximity weight: increases exponentially as distance to floor approaches 0
        proximity = np.exp(-np.maximum(0.0, distances) / self.y_crash_zone)
        # Asymmetry: if predicted current < true current when low, it's dangerous (under-powering)
        underpower = (y_pred < y_true).astype(np.float64)
        penalty_multiplier = 1.0 + self.lambda_crash * proximity * (1.0 + self.gamma_underpower * underpower)
        return penalty_multiplier

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray, distances: np.ndarray) -> float:
        w = self.compute_weights(y_pred, y_true, distances)
        return float(np.mean(w * ((y_pred - y_true) ** 2)))

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, distances: np.ndarray) -> np.ndarray:
        w = self.compute_weights(y_pred, y_true, distances)
        return 2.0 * w * (y_pred - y_true)


# =====================================================================
# Feature Standardizer
# =====================================================================

class FeatureStandardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)
        self.std[self.std < 1e-8] = 1.0

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / self.std

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


# =====================================================================
# Multi-Layer Perceptron (Neural Network Controller)
# =====================================================================

class AntigravityNeuralNetwork:
    """
    Multi-Layer Perceptron (MLP) Controller for Magnetic Levitation.
    Input: [Distance y, Velocity v]
    Output: [Predicted Electromagnet Current I]
    """

    def __init__(
        self,
        hidden_sizes: list[int] = (16, 16),
        activation: str = "relu",
        seed: int = 42,
    ):
        self.hidden_sizes = list(hidden_sizes)
        self.activation = activation
        self.seed = seed
        self.scaler = FeatureStandardizer()

        # Build architecture: 2 inputs -> hidden layers -> 1 output (linear)
        self.layers: list[DenseLayer] = []
        in_dim = 2
        for i, h_dim in enumerate(self.hidden_sizes):
            self.layers.append(
                DenseLayer(in_dim, h_dim, activation=self.activation, seed=self.seed + i if self.seed else None)
            )
            in_dim = h_dim
        # Output layer: 1 neuron with linear activation
        self.layers.append(
            DenseLayer(in_dim, 1, activation="linear", seed=self.seed + 999 if self.seed else None)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward propagation through all layers.
        x: normalized input batch, shape (N, 2)
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, d_loss: np.ndarray):
        """
        Backward propagation through all layers in reverse.
        d_loss: gradient w.r.t network output, shape (N, 1)
        """
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Inference on physical state [y, v].
        raw_state can be shape (2,) or (N, 2).
        Returns predicted current clamped to [0, max_current].
        """
        arr = np.asarray(raw_state, dtype=np.float64)
        single = (arr.ndim == 1)
        if single:
            arr = arr.reshape(1, -1)

        x_norm = self.scaler.transform(arr)
        pred = self.forward(x_norm)
        # Electrical current must be non-negative
        pred = np.maximum(0.0, pred)

        if single:
            return float(pred[0, 0])
        return pred

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        optimizer_type: str = "adam",
        learning_rate: float = 0.005,
        batch_size: int = 64,
        epochs: int = 150,
        loss_type: str = "mse",
        crash_penalty_params: dict = None,
        verbose: bool = True,
    ) -> dict:
        """
        Train the MLP using mini-batch gradient descent.
        """
        # Fit normalizer on training inputs
        X_train_norm = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_norm = self.scaler.transform(X_val)

        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)

        # Initialize optimizer
        if optimizer_type.lower() == "adam":
            optimizer = AdamOptimizer(lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = SGDOptimizer(lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Initialize loss function
        if loss_type.lower() == "crash_penalty":
            params = crash_penalty_params or {}
            criterion = CrashPenaltyLoss(**params)
        else:
            criterion = MSELoss()

        history = {"train_loss": [], "val_loss": [], "mse_loss": []}
        N = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle mini-batches
            indices = np.random.permutation(N)
            epoch_losses = []

            for start_idx in range(0, N, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                X_batch_norm = X_train_norm[batch_idx]
                X_batch_raw = X_train[batch_idx]
                y_batch = y_train[batch_idx]

                # Forward pass
                pred = self.forward(X_batch_norm)

                # Compute loss and gradient
                if loss_type.lower() == "crash_penalty":
                    distances = X_batch_raw[:, 0:1]  # distance/height column
                    loss_val = criterion.forward(pred, y_batch, distances)
                    d_loss = criterion.backward(pred, y_batch, distances)
                else:
                    loss_val = criterion.forward(pred, y_batch)
                    d_loss = criterion.backward(pred, y_batch)

                epoch_losses.append(loss_val)

                # Backward pass
                self.backward(d_loss)

                # Optimizer step
                optimizer.step(self.layers)

            avg_train_loss = float(np.mean(epoch_losses))
            history["train_loss"].append(avg_train_loss)

            # Evaluate on validation set
            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val_norm)
                val_mse = MSELoss.forward(val_pred, y_val)
                history["val_loss"].append(val_mse)
                history["mse_loss"].append(val_mse)

            if verbose and (epoch == 1 or epoch % 25 == 0 or epoch == epochs):
                val_info = f" | Val MSE: {history['val_loss'][-1]:.6f}" if X_val is not None else ""
                print(f"Epoch [{epoch:3d}/{epochs:3d}] - Loss ({loss_type}): {avg_train_loss:.6f}{val_info}")

        return history

    # =================================================================
    # Persistence: Joblib & Pickle
    # =================================================================

    def save(self, filepath: str, use_joblib: bool = True):
        """
        Save trained model weights, architecture, and scaler to file.
        """
        state = {
            "hidden_sizes": self.hidden_sizes,
            "activation": self.activation,
            "seed": self.seed,
            "scaler_mean": self.scaler.mean,
            "scaler_std": self.scaler.std,
            "weights": [layer.W.copy() for layer in self.layers],
            "biases": [layer.b.copy() for layer in self.layers],
        }
        if use_joblib and HAS_JOBLIB:
            joblib.dump(state, filepath)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(state, f)
        print(f"[Model Saved] Saved weights and metadata to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AntigravityNeuralNetwork":
        """
        Load model weights, architecture, and scaler from file.
        """
        if HAS_JOBLIB:
            try:
                state = joblib.load(filepath)
            except Exception:
                with open(filepath, "rb") as f:
                    state = pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                state = pickle.load(f)

        model = cls(
            hidden_sizes=state["hidden_sizes"],
            activation=state["activation"],
            seed=state.get("seed", 42),
        )
        model.scaler.mean = state["scaler_mean"]
        model.scaler.std = state["scaler_std"]

        for i, layer in enumerate(model.layers):
            layer.W = state["weights"][i].copy()
            layer.b = state["biases"][i].copy()

        print(f"[Model Loaded] Successfully loaded controller from: {filepath}")
        return model
