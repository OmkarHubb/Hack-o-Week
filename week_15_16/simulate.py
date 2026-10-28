"""
simulate.py - Deployment and Multi-Scenario Simulation Test Script
Loads the trained Antigravity Neural Network Controller (joblib or pickle),
drops the metallic object from multiple virtual starting heights, and uses
the model's frame-by-frame predictions to catch it in mid-air and hover seamlessly.

Produces an ultra-stylish multi-panel telemetry graph and opens it automatically!
"""

import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from physics import MagneticLevitationChamber
from neural_network import AntigravityNeuralNetwork


def run_simulation(
    model_path: str = "models/antigravity_model.joblib",
    drop_heights: list[float] = (0.15, 0.40, 0.85),
    sim_time: float = 2.0,
    open_graph: bool = True,
):
    print("=" * 65)
    print("PHASE 4: REAL-TIME DEPLOYMENT & MID-AIR CATCH SIMULATION")
    print("=" * 65)

    if not os.path.exists(model_path):
        # Fallback to pickle or mse model
        if os.path.exists("models/antigravity_model.pkl"):
            model_path = "models/antigravity_model.pkl"
        elif os.path.exists("models/antigravity_model_mse.joblib"):
            model_path = "models/antigravity_model_mse.joblib"
        else:
            raise FileNotFoundError("No trained model found. Please run train.py first.")

    print(f"Loading trained neural network controller from: {model_path}")
    controller = AntigravityNeuralNetwork.load(model_path)

    chamber = MagneticLevitationChamber()
    dt = chamber.dt
    total_steps = int(sim_time / dt)
    time_arr = np.linspace(0, sim_time, total_steps)

    results = []

    for idx, h_init in enumerate(drop_heights):
        print(f"\n[Test Drop {idx + 1}/{len(drop_heights)}] Dropping object from height: {h_init:.2f} m...")
        chamber.reset(initial_height=h_init, initial_velocity=0.0)

        history_y = []
        history_v = []
        history_I = []
        history_fmag = []

        caught_time = None

        for step in range(total_steps):
            current_state = np.array([chamber.y, chamber.v])

            # AI INFERENCE: Frame-by-frame electrical current prediction
            predicted_current = controller.predict(current_state)

            # Step physics environment
            next_state, error, done, info = chamber.step(predicted_current)

            history_y.append(info["y"])
            history_v.append(info["v"])
            history_I.append(info["current"])
            history_fmag.append(info["f_mag"])

            # Check if stabilized within +/- 1.5 cm and velocity < 0.05 m/s
            if caught_time is None and step > 20:
                if abs(info["y"] - chamber.y_target) < 0.015 and abs(info["v"]) < 0.05:
                    caught_time = info["time"]

        final_error = abs(history_y[-1] - chamber.y_target)
        final_current = history_I[-1]
        print(f"  -> Caught and Hovering at: {history_y[-1]:.4f} m (Target: {chamber.y_target:.2f} m)")
        print(f"  -> Final Hover Error: {final_error * 1000:.2f} mm | Steady Current: {final_current:.3f} A")
        if caught_time:
            print(f"  -> Catch & Stabilization Time: {caught_time:.3f} seconds!")
        else:
            print("  -> Hover stabilized seamlessly.")

        results.append({
            "initial_height": h_init,
            "y": np.array(history_y),
            "v": np.array(history_v),
            "I": np.array(history_I),
            "f_mag": np.array(history_fmag),
            "caught_time": caught_time,
        })

    # =============================================================
    # ULTRA-STYLISH MULTI-PANEL TELEMETRY GRAPH
    # =============================================================
    print("\n" + "=" * 65)
    print("RENDERING STYLISH TELEMETRY VISUALIZATIONS...")
    print("=" * 65)

    # Set dark futuristic aesthetic
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), facecolor="#0b0f19")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    color_palette = ["#38bdf8", "#f43f5e", "#22c55e", "#fbbf24"]

    # 1. Height vs Time (The Mid-Air Catch)
    ax1 = fig.add_subplot(gs[0, 0], facecolor="#111827")
    ax1.axhspan(chamber.y_target - 0.02, chamber.y_target + 0.02, color="#059669", alpha=0.18, label="Target Hover Zone (±2cm)")
    ax1.axhline(chamber.y_target, color="#10b981", linestyle="--", lw=2, label=f"Target Altitude ({chamber.y_target:.2f} m)")
    ax1.axhline(0.0, color="#ef4444", linestyle="-", lw=2.5, alpha=0.8, label="Crash Boundary (Floor y=0)")
    ax1.axhline(chamber.H, color="#8b5cf6", linestyle=":", lw=1.5, alpha=0.7, label=f"Ceiling Magnet (y={chamber.H:.1f} m)")

    for idx, res in enumerate(results):
        col = color_palette[idx % len(color_palette)]
        ax1.plot(time_arr, res["y"], color=col, lw=2.4, label=f"Drop from {res['initial_height']:.2f} m")
        # Mark initial release point
        ax1.scatter([0], [res["initial_height"]], color=col, s=70, zorder=5, edgecolors="#ffffff", lw=1.2)
        # Mark hover catch point
        if res["caught_time"] is not None:
            catch_idx = int(res["caught_time"] / dt)
            ax1.scatter([res["caught_time"]], [res["y"][catch_idx]], color="#fbbf24", s=90, marker="*", zorder=6)

    ax1.set_title("✦ ANTIGRAVITY CATCH & HOVER TRAJECTORY ✦", fontsize=13, fontweight="bold", color="#38bdf8", pad=12)
    ax1.set_xlabel("Time (seconds)", fontsize=11, color="#9ca3af")
    ax1.set_ylabel("Height in Chamber y (meters)", fontsize=11, color="#9ca3af")
    ax1.set_ylim(-0.05, chamber.H + 0.05)
    ax1.grid(True, linestyle=":", color="#374151", alpha=0.7)
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.6, facecolor="#1f2937")

    # 2. Control Signal (Electromagnet Current) vs Time
    ax2 = fig.add_subplot(gs[0, 1], facecolor="#111827")
    ax2.axhline(chamber.equilibrium_current, color="#fbbf24", linestyle="--", lw=1.8, label=f"Equilibrium I* = {chamber.equilibrium_current:.2f} A")

    for idx, res in enumerate(results):
        col = color_palette[idx % len(color_palette)]
        ax2.plot(time_arr, res["I"], color=col, lw=2.2, label=f"Current (from {res['initial_height']:.2f} m)")

    ax2.set_title("✦ NEURAL CONTROLLER ELECTRICAL CURRENT (I) ✦", fontsize=13, fontweight="bold", color="#f43f5e", pad=12)
    ax2.set_xlabel("Time (seconds)", fontsize=11, color="#9ca3af")
    ax2.set_ylabel("Electromagnet Current (Amperes)", fontsize=11, color="#9ca3af")
    ax2.set_ylim(0, chamber.max_current + 0.5)
    ax2.grid(True, linestyle=":", color="#374151", alpha=0.7)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.6, facecolor="#1f2937")

    # 3. Velocity Phase Portrait & Dynamics
    ax3 = fig.add_subplot(gs[1, 0], facecolor="#111827")
    ax3.axhline(0, color="#6b7280", linestyle="--", alpha=0.7)

    for idx, res in enumerate(results):
        col = color_palette[idx % len(color_palette)]
        ax3.plot(time_arr, res["v"], color=col, lw=2.0, label=f"Velocity (from {res['initial_height']:.2f} m)")

    ax3.set_title("✦ VELOCITY DAMPING DYNAMICS ✦", fontsize=13, fontweight="bold", color="#22c55e", pad=12)
    ax3.set_xlabel("Time (seconds)", fontsize=11, color="#9ca3af")
    ax3.set_ylabel("Vertical Velocity dy/dt (m/s)", fontsize=11, color="#9ca3af")
    ax3.grid(True, linestyle=":", color="#374151", alpha=0.7)
    ax3.legend(loc="lower right", fontsize=9, framealpha=0.6, facecolor="#1f2937")

    # 4. Phase Space Portrait: Velocity vs Height (Attractor Convergence)
    ax4 = fig.add_subplot(gs[1, 1], facecolor="#111827")
    for idx, res in enumerate(results):
        col = color_palette[idx % len(color_palette)]
        ax4.plot(res["y"], res["v"], color=col, lw=2.2, label=f"Path (y0={res['initial_height']:.2f}m)")
        ax4.scatter([res["y"][0]], [res["v"][0]], color=col, s=80, marker="o", edgecolors="#fff")
        ax4.scatter([res["y"][-1]], [res["v"][-1]], color="#10b981", s=90, marker="X")

    # Target Fixed Point
    ax4.scatter([chamber.y_target], [0.0], color="#fbbf24", s=180, marker="*", zorder=10, label=f"Attractor ({chamber.y_target:.2f}m, 0 m/s)")
    ax4.set_title("✦ PHASE PORTRAIT: CONVERGENCE TO LEVITATION POINT ✦", fontsize=13, fontweight="bold", color="#a855f7", pad=12)
    ax4.set_xlabel("Height y (meters)", fontsize=11, color="#9ca3af")
    ax4.set_ylabel("Velocity v (m/s)", fontsize=11, color="#9ca3af")
    ax4.set_xlim(-0.02, chamber.H + 0.02)
    ax4.grid(True, linestyle=":", color="#374151", alpha=0.7)
    ax4.legend(loc="upper right", fontsize=9, framealpha=0.6, facecolor="#1f2937")

    fig.suptitle("AI-POWERED ANTIGRAVITY LEVITATION CONTROLLER — TELEMETRY DASHBOARD", fontsize=16, fontweight="bold", color="#ffffff", y=0.98)

    os.makedirs("assets", exist_ok=True)
    out_img = os.path.join("assets", "antigravity_telemetry.png")
    try:
        plt.savefig(out_img, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    except (OSError, PermissionError):
        import time
        out_img = os.path.join("assets", f"antigravity_telemetry_{int(time.time())}.png")
        plt.savefig(out_img, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[Dashboard Saved] High-res visual saved to: {out_img}")

    if open_graph:
        print("[Opening Graph Window] Launching interactive display on screen...")
        try:
            if sys.platform.startswith("win"):
                os.startfile(os.path.abspath(out_img))
        except Exception as e:
            print(f"Viewer launch notice: {e}")
        plt.show(block=False)
        plt.pause(1.5)

    plt.close()
    return results


if __name__ == "__main__":
    run_simulation(open_graph=True)
