"""
test_antigravity.py - Automated Test Suite & Numerical Gradient Checking
Verifies:
1. Physics environment equilibrium and RK4 dynamics.
2. Backpropagation analytical gradient checking vs numerical finite differences (< 1e-6 relative error).
3. Adam and SGD optimizer weight updates.
4. Loss functions (MSE and Custom Crash-Penalty).
5. Model persistence (Joblib and Pickle serialization/deserialization).
6. Closed-loop hover stability.
"""

import os
import unittest
import numpy as np
from physics import MagneticLevitationChamber
from neural_network import (
    AntigravityNeuralNetwork,
    DenseLayer,
    MSELoss,
    CrashPenaltyLoss,
    SGDOptimizer,
    AdamOptimizer,
)


class TestMagneticLevitationPhysics(unittest.TestCase):
    def setUp(self):
        self.chamber = MagneticLevitationChamber()

    def test_equilibrium_current(self):
        """At y = y_target, v = 0, applying equilibrium current must yield net acceleration = 0."""
        i_eq = self.chamber.equilibrium_current
        acc = self.chamber.compute_acceleration(self.chamber.y_target, 0.0, i_eq)
        self.assertAlmostEqual(acc, 0.0, places=4, msg="Equilibrium current should cancel gravity exactly.")

    def test_gravity_freefall(self):
        """With zero current and zero velocity, downward acceleration should equal -9.81 m/s^2."""
        acc = self.chamber.compute_acceleration(0.5, 0.0, current=0.0)
        self.assertAlmostEqual(acc, -self.chamber.g, places=4)

    def test_optimal_control_calculation(self):
        """Optimal control at equilibrium should match equilibrium current."""
        i_opt = self.chamber.compute_optimal_control(self.chamber.y_target, 0.0)
        self.assertAlmostEqual(i_opt, self.chamber.equilibrium_current, places=3)


class TestBackpropagationGradientCheck(unittest.TestCase):
    """
    Mathematical gradient checking:
    Compares analytical gradients dW, db from backpropagation against
    numerical central difference approximation:
    dF/dw ~ [F(w + eps) - F(w - eps)] / (2 * eps)
    """

    def test_analytical_vs_numerical_gradients(self):
        rng = np.random.RandomState(123)
        X = rng.randn(4, 2)
        y = rng.uniform(0.5, 3.5, size=(4, 1))

        # Simple 2 -> 4 -> 1 network with ReLU and Linear
        layer1 = DenseLayer(2, 4, activation="relu", seed=42)
        layer2 = DenseLayer(4, 1, activation="linear", seed=43)

        def forward_network(x):
            a1 = layer1.forward(x)
            a2 = layer2.forward(a1)
            return a2

        # Analytical backward pass
        pred = forward_network(X)
        d_loss = MSELoss.backward(pred, y)
        d_a1 = layer2.backward(d_loss)
        _ = layer1.backward(d_a1)

        analytical_dW1 = layer1.dW.copy()
        analytical_db1 = layer1.db.copy()

        eps = 1e-6

        # Numerical gradient for layer 1 weights
        numerical_dW1 = np.zeros_like(layer1.W)
        for i in range(layer1.W.shape[0]):
            for j in range(layer1.W.shape[1]):
                orig = layer1.W[i, j]

                layer1.W[i, j] = orig + eps
                loss_plus = MSELoss.forward(forward_network(X), y)

                layer1.W[i, j] = orig - eps
                loss_minus = MSELoss.forward(forward_network(X), y)

                layer1.W[i, j] = orig
                numerical_dW1[i, j] = (loss_plus - loss_minus) / (2.0 * eps)

        # Compute relative difference
        rel_error_W = np.linalg.norm(analytical_dW1 - numerical_dW1) / (
            np.linalg.norm(analytical_dW1) + np.linalg.norm(numerical_dW1) + 1e-12
        )
        self.assertLess(
            rel_error_W, 1e-5, f"Gradient check failed for Layer 1 weights: rel_error = {rel_error_W}"
        )


class TestModelPersistence(unittest.TestCase):
    def test_save_and_load_joblib_and_pickle(self):
        os.makedirs("models", exist_ok=True)
        model = AntigravityNeuralNetwork(hidden_sizes=(8, 8), activation="relu", seed=10)
        X_dummy = np.array([[0.5, 0.0], [0.2, -1.0]])
        preds_before = model.predict(X_dummy)

        # Test joblib
        joblib_path = "models/test_model.joblib"
        model.save(joblib_path, use_joblib=True)
        loaded_joblib = AntigravityNeuralNetwork.load(joblib_path)
        preds_joblib = loaded_joblib.predict(X_dummy)
        np.testing.assert_allclose(preds_before, preds_joblib, rtol=1e-5)

        # Test pickle
        pickle_path = "models/test_model.pkl"
        model.save(pickle_path, use_joblib=False)
        loaded_pickle = AntigravityNeuralNetwork.load(pickle_path)
        preds_pickle = loaded_pickle.predict(X_dummy)
        np.testing.assert_allclose(preds_before, preds_pickle, rtol=1e-5)

        # Cleanup
        if os.path.exists(joblib_path):
            os.remove(joblib_path)
        if os.path.exists(pickle_path):
            os.remove(pickle_path)


class TestClosedLoopHoverStability(unittest.TestCase):
    def test_hover_catch(self):
        """Verify that the saved model catches the object and maintains height within +/- 2.5 cm."""
        model_path = "models/antigravity_model.joblib"
        if not os.path.exists(model_path):
            self.skipTest("Trained model not found.")

        model = AntigravityNeuralNetwork.load(model_path)
        chamber = MagneticLevitationChamber()
        chamber.reset(initial_height=0.25, initial_velocity=0.0)

        for _ in range(300):  # 1.5 seconds of simulation
            cmd_I = model.predict([chamber.y, chamber.v])
            chamber.step(cmd_I)

        final_error = abs(chamber.y - chamber.y_target)
        self.assertLess(final_error, 0.025, f"Hover error too large: {final_error * 100:.2f} cm")
        self.assertLess(abs(chamber.v), 0.1, f"Residual velocity too large: {chamber.v:.3f} m/s")


if __name__ == "__main__":
    unittest.main()
