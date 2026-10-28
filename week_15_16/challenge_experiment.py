"""
challenge_experiment.py - User Challenge: Crash Penalty vs Standard MSE Benchmark
Directly addresses the prompt's challenge:
"Can you implement a custom loss function that penalizes the model more if the
object 'crashes' (hits distance = 0) than if it just wobbles slightly in the air?"

Compares the standard MSE neural controller against the custom crash-penalty controller
under extreme low-altitude drop scenarios with high downward velocities.
Generates comparative graphs and opens the window automatically!
"""

import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
from physics import MagneticLevitationChamber
from neural_network import AntigravityNeuralNetwork


def run_challenge_benchmark(open_graph: bool = True):
    print("=" * 70)
    print("USER CHALLENGE BENCHMARK: CRASH PENALTY VS STANDARD MSE CONTROLLER")
    print("=" * 70)

    model_mse_path = "models/antigravity_model_mse.joblib"
    model_crash_path = "models/antigravity_model_crash_penalty.joblib"

    if not os.path.exists(model_mse_path) or not os.path.exists(model_crash_path):
        raise FileNotFoundError("Both models must be trained first. Please run train.py.")

    print(f"Loading Model A (Standard MSE):       {model_mse_path}")
    print(f"Loading Model B (Custom Crash-Loss):   {model_crash_path}")

    model_mse = AntigravityNeuralNetwork.load(model_mse_path)
    model_crash = AntigravityNeuralNetwork.load(model_crash_path)

    # Test cases: Extreme low-altitude drops with downward momentum
    scenarios = [
        {"name": "Hazard Drop 1 (Low Release)", "y0": 0.10, "v0": -0.8, "duration": 1.8},
        {"name": "Hazard Drop 2 (Near-Floor Release)", "y0": 0.06, "v0": -0.6, "duration": 1.8},
        {"name": "Hazard Drop 3 (High Velocity Dive)", "y0": 0.15, "v0": -2.2, "duration": 1.8},
    ]

    chamber = MagneticLevitationChamber()
    dt = chamber.dt

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(16, 4.2 * len(scenarios)), facecolor="#090d16")
    plt.style.use("dark_background")

    for s_idx, sc in enumerate(scenarios):
        y0, v0, dur = sc["y0"], sc["v0"], sc["duration"]
        steps = int(dur / dt)
        times = np.linspace(0, dur, steps)

        # Simulation 1: Standard MSE Controller
        chamber.reset(initial_height=y0, initial_velocity=v0)
        y_mse_hist = []
        i_mse_hist = []
        crashed_mse = False
        min_y_mse = y0

        for _ in range(steps):
            pred_I = model_mse.predict([chamber.y, chamber.v])
            _, _, _, info = chamber.step(pred_I)
            y_mse_hist.append(info["y"])
            i_mse_hist.append(info["current"])
            min_y_mse = min(min_y_mse, info["y"])
            if info["crashed"]:
                crashed_mse = True

        # Simulation 2: Crash-Penalty Controller
        chamber.reset(initial_height=y0, initial_velocity=v0)
        y_crash_hist = []
        i_crash_hist = []
        crashed_crash = False
        min_y_crash = y0

        for _ in range(steps):
            pred_I = model_crash.predict([chamber.y, chamber.v])
            _, _, _, info = chamber.step(pred_I)
            y_crash_hist.append(info["y"])
            i_crash_hist.append(info["current"])
            min_y_crash = min(min_y_crash, info["y"])
            if info["crashed"]:
                crashed_crash = True

        print(f"\nScenario [{s_idx + 1}]: {sc['name']} (y0 = {y0:.2f}m, v0 = {v0:+.2f}m/s)")
        print(f"  Standard MSE Model:  Min Height = {min_y_mse * 100:.1f} cm | Crashed = {crashed_mse}")
        print(f"  Crash-Penalty Model: Min Height = {min_y_crash * 100:.1f} cm | Crashed = {crashed_crash}")

        # Plot Subplot Left: Trajectory (Height)
        ax_y = axes[s_idx, 0]
        ax_y.set_facecolor("#0e1726")
        ax_y.axhline(chamber.y_target, color="#10b981", linestyle="--", lw=1.5, alpha=0.8, label="Target (0.50m)")
        ax_y.axhline(0.0, color="#ef4444", lw=2.2, label="Floor (CRASH y=0)")
        ax_y.plot(times, y_mse_hist, color="#38bdf8", lw=2.2, label=f"Standard MSE (Min: {min_y_mse*100:.1f}cm)")
        ax_y.plot(times, y_crash_hist, color="#f59e0b", lw=2.4, label=f"Crash-Penalty Loss (Min: {min_y_crash*100:.1f}cm)")

        ax_y.set_title(f"{sc['name']} — Height vs Time", fontsize=11, fontweight="bold", color="#e2e8f0")
        ax_y.set_ylabel("Height (m)", fontsize=10, color="#9ca3af")
        ax_y.set_ylim(-0.04, 0.65)
        ax_y.grid(True, linestyle=":", color="#374151", alpha=0.6)
        ax_y.legend(loc="lower right", fontsize=8.5, facecolor="#1f2937", framealpha=0.7)

        # Plot Subplot Right: Electromagnet Current
        ax_i = axes[s_idx, 1]
        ax_i.set_facecolor("#0e1726")
        ax_i.plot(times, i_mse_hist, color="#38bdf8", lw=2.0, label="Standard MSE Current")
        ax_i.plot(times, i_crash_hist, color="#f59e0b", lw=2.2, label="Crash-Penalty Current")
        ax_i.axhline(chamber.equilibrium_current, color="#34d399", linestyle=":", lw=1.5, label="Equilibrium I*")

        ax_i.set_title(f"{sc['name']} — Control Current I(t)", fontsize=11, fontweight="bold", color="#e2e8f0")
        ax_i.set_ylabel("Current (A)", fontsize=10, color="#9ca3af")
        ax_i.set_ylim(0, chamber.max_current + 0.5)
        ax_i.grid(True, linestyle=":", color="#374151", alpha=0.6)
        ax_i.legend(loc="upper right", fontsize=8.5, facecolor="#1f2937", framealpha=0.7)

    fig.suptitle("USER CHALLENGE: STANDARD MSE VS CUSTOM CRASH-PENALTY LOSS", fontsize=14, fontweight="bold", color="#ffffff", y=0.995)
    plt.tight_layout()

    os.makedirs("assets", exist_ok=True)
    out_path = os.path.join("assets", "challenge_crash_comparison.png")
    try:
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    except (OSError, PermissionError):
        import time
        out_path = os.path.join("assets", f"challenge_crash_comparison_{int(time.time())}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n[Chart Saved] Challenge benchmark graph saved to: {out_path}")

    if open_graph:
        print("[Opening Graph Window] Launching comparison graph display on screen...")
        try:
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(out_path))
        except Exception as e:
            print(f"Viewer launch notice: {e}")
        plt.show(block=False)
        plt.pause(1.5)

    plt.close()
    return out_path


if __name__ == "__main__":
    run_challenge_benchmark(open_graph=True)
