"""
animate.py - Ultra-Stylish Antigravity Chamber Real-Time Animation
Simulates a metallic sphere dropped from an unstable height and shows the
AI Neural Network Controller catching it mid-air and holding it suspended.

Features:
- Dark Futuristic Laboratory / Cyberpunk Visual Aesthetic.
- Dynamic glowing electromagnet flux field that scales with commanded current.
- Metallic levitating sphere with specular shine and floor contact shadow.
- Force balance vectors: Upward Magnetic Pull (F_mag) vs Downward Gravity (F_g).
- Laser height sensor beam with real-time distance targeting dot.
- Live telemetry oscilloscope HUD for Height, Current, and Velocity.
- Generates high-FPS animated GIF and opens interactive display window!
"""

import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from physics import MagneticLevitationChamber
from neural_network import AntigravityNeuralNetwork


def create_antigravity_animation(
    model_path: str = "models/antigravity_model.joblib",
    drop_height: float = 0.18,  # Start dangerously low to show dramatic mid-air catch!
    duration: float = 2.5,
    fps: int = 40,
    save_gif: bool = True,
    show_window: bool = True,
):
    print("=" * 65)
    print("INITIALIZING STYLISH ANTIGRAVITY ANIMATION...")
    print(f"Drop Height: {drop_height:.2f} m | Duration: {duration} s | Target: 0.50 m")
    print("=" * 65)

    if not os.path.exists(model_path):
        if os.path.exists("models/antigravity_model.pkl"):
            model_path = "models/antigravity_model.pkl"
        elif os.path.exists("models/antigravity_model_mse.joblib"):
            model_path = "models/antigravity_model_mse.joblib"
        else:
            raise FileNotFoundError("Model not found. Run train.py first.")

    controller = AntigravityNeuralNetwork.load(model_path)
    chamber = MagneticLevitationChamber()

    # Pre-simulate trajectory at high physics rate (200 Hz)
    dt_sim = chamber.dt
    total_sim_steps = int(duration / dt_sim)

    sim_times = []
    sim_y = []
    sim_v = []
    sim_I = []
    sim_fmag = []

    chamber.reset(initial_height=drop_height, initial_velocity=0.0)

    for step in range(total_sim_steps):
        state = np.array([chamber.y, chamber.v])
        current = controller.predict(state)
        _, _, _, info = chamber.step(current)

        sim_times.append(info["time"])
        sim_y.append(info["y"])
        sim_v.append(info["v"])
        sim_I.append(info["current"])
        sim_fmag.append(info["f_mag"])

    sim_times = np.array(sim_times)
    sim_y = np.array(sim_y)
    sim_v = np.array(sim_v)
    sim_I = np.array(sim_I)
    sim_fmag = np.array(sim_fmag)

    # Subsample for animation frames
    n_frames = int(duration * fps)
    frame_indices = np.linspace(0, total_sim_steps - 1, n_frames).astype(int)

    # =============================================================
    # FIGURE & LAYOUT SETUP
    # =============================================================
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(15, 8.5), facecolor="#090d16")

    # Grid: Left = Chamber View (col 0, row 0:3), Right = 3 Oscilloscopes (col 1, rows 0, 1, 2)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1.4], wspace=0.28, hspace=0.35)

    ax_chamber = fig.add_subplot(gs[:, 0], facecolor="#0d131f")
    ax_y = fig.add_subplot(gs[0, 1], facecolor="#0d131f")
    ax_I = fig.add_subplot(gs[1, 1], facecolor="#0d131f")
    ax_v = fig.add_subplot(gs[2, 1], facecolor="#0d131f")

    # -------------------------------------------------------------
    # CHAMBER STATIC ARTWORK
    # -------------------------------------------------------------
    ax_chamber.set_xlim(-0.6, 0.6)
    ax_chamber.set_ylim(-0.08, chamber.H + 0.15)
    ax_chamber.set_xticks([])
    ax_chamber.set_yticks(np.arange(0.0, chamber.H + 0.1, 0.1))
    ax_chamber.tick_params(colors="#6b7280", labelsize=9)
    ax_chamber.grid(True, linestyle=":", color="#1f2937", alpha=0.6, axis="y")
    ax_chamber.set_ylabel("Chamber Altitude (meters)", color="#9ca3af", fontsize=11, labelpad=8)
    ax_chamber.set_title("✦ ANTIGRAVITY MAGNETIC CHAMBER ✦", color="#38bdf8", fontsize=13, fontweight="bold", pad=12)

    # Chamber Glass Enclosure Walls
    chamber_box = patches.Rectangle(
        (-0.45, 0.0), 0.9, chamber.H,
        linewidth=2, edgecolor="#38bdf8", facecolor="#0e1726", alpha=0.35, zorder=1
    )
    ax_chamber.add_patch(chamber_box)

    # Crash Zone Hazard Stripes Floor (y = 0)
    floor_box = patches.Rectangle((-0.55, -0.06), 1.1, 0.06, linewidth=0, facecolor="#7f1d1d", alpha=0.7, zorder=2)
    ax_chamber.add_patch(floor_box)
    ax_chamber.text(0, -0.035, "▲ CRASH BOUNDARY HAZARD ZONE ▲", color="#fca5a5", fontsize=9, fontweight="bold", ha="center", va="center")

    # Electromagnet Casing at Ceiling (y = H)
    magnet_case = patches.Rectangle((-0.3, chamber.H), 0.6, 0.1, linewidth=1.5, edgecolor="#8b5cf6", facecolor="#2e1065", zorder=3)
    ax_chamber.add_patch(magnet_case)
    # Coil turns visual
    for cx in np.linspace(-0.24, 0.24, 9):
        coil = patches.Circle((cx, chamber.H + 0.05), 0.03, facecolor="#c084fc", edgecolor="#e9d5ff", alpha=0.85, zorder=4)
        ax_chamber.add_patch(coil)
    ax_chamber.text(0, chamber.H + 0.11, "ELECTROMAGNET SOLENOID", color="#c084fc", fontsize=8, fontweight="bold", ha="center")

    # Target Height Guideline & Band
    ax_chamber.axhspan(chamber.y_target - 0.015, chamber.y_target + 0.015, color="#059669", alpha=0.22, zorder=2)
    ax_chamber.axhline(chamber.y_target, color="#10b981", linestyle="--", lw=1.8, alpha=0.9, zorder=2)
    ax_chamber.text(0.35, chamber.y_target + 0.02, "TARGET: 0.50m", color="#34d399", fontsize=9, fontweight="bold", ha="center")

    # Dynamic Magnetic Flux Field Glow (Ellipse whose alpha and size pulse with current)
    flux_glow = patches.Ellipse((0, chamber.H), 0.6, 0.4, color="#a855f7", alpha=0.1, zorder=2)
    ax_chamber.add_patch(flux_glow)

    # Dynamic Laser Beam (thin line from ceiling to ball)
    laser_line, = ax_chamber.plot([0, 0], [chamber.H, drop_height], color="#ef4444", lw=1.5, alpha=0.75, linestyle=":")
    laser_dot, = ax_chamber.plot([0], [drop_height], marker="o", markersize=5, color="#f87171", alpha=0.9)

    # Metallic Sphere (Main Ball)
    ball_radius = 0.035
    ball_patch = patches.Circle((0, drop_height), ball_radius, facecolor="#94a3b8", edgecolor="#e2e8f0", lw=2, zorder=6)
    ax_chamber.add_patch(ball_patch)

    # Specular highlight on sphere to give real 3D metallic sphere look
    shine_patch = patches.Circle((0.008, drop_height + 0.012), 0.01, facecolor="#ffffff", alpha=0.85, lw=0, zorder=7)
    ax_chamber.add_patch(shine_patch)

    # Ground Contact Shadow (grows darker and larger as ball nears floor)
    shadow_patch = patches.Ellipse((0, 0.005), 0.08, 0.012, color="#000000", alpha=0.6, zorder=2)
    ax_chamber.add_patch(shadow_patch)

    # Force Vectors (Upward Magnetic Force, Downward Gravity)
    arrow_fmag = ax_chamber.annotate(
        "", xy=(0, drop_height + 0.08), xytext=(0, drop_height),
        arrowprops=dict(arrowstyle="->", color="#c084fc", lw=2.5, mutation_scale=15),
        zorder=8
    )
    arrow_fgrav = ax_chamber.annotate(
        "", xy=(0, drop_height - 0.08), xytext=(0, drop_height),
        arrowprops=dict(arrowstyle="->", color="#f87171", lw=2.0, mutation_scale=15),
        zorder=8
    )

    # Chamber Live HUD Text
    hud_text = ax_chamber.text(
        -0.42, 0.06, "",
        fontsize=9, fontfamily="monospace", color="#e0e7ff",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#0b1120", edgecolor="#38bdf8", alpha=0.88, lw=1.2),
        zorder=9
    )

    # -------------------------------------------------------------
    # OSCILLOSCOPE SUBPLOTS (Live Telemetry Graphs)
    # -------------------------------------------------------------
    for ax in (ax_y, ax_I, ax_v):
        ax.grid(True, linestyle=":", color="#374151", alpha=0.6)
        ax.tick_params(colors="#9ca3af", labelsize=8.5)

    # Ax 1: Altitude y(t)
    ax_y.set_xlim(0, duration)
    ax_y.set_ylim(-0.02, chamber.H + 0.02)
    ax_y.axhline(chamber.y_target, color="#10b981", linestyle="--", lw=1.5, alpha=0.8, label=f"Target ({chamber.y_target}m)")
    line_y, = ax_y.plot([], [], color="#38bdf8", lw=2.2, label="Sphere Height")
    dot_y, = ax_y.plot([], [], marker="o", color="#38bdf8", markersize=6)
    ax_y.set_ylabel("Height y (m)", color="#38bdf8", fontsize=9.5, fontweight="bold")
    ax_y.set_title("LIVE TELEMETRY: ALTITUDE TRACKING", color="#e2e8f0", fontsize=10.5, fontweight="bold", pad=6)
    ax_y.legend(loc="upper right", fontsize=8, facecolor="#1f2937", framealpha=0.6)

    # Ax 2: Control Current I(t)
    ax_I.set_xlim(0, duration)
    ax_I.set_ylim(0, chamber.max_current + 0.5)
    ax_I.axhline(chamber.equilibrium_current, color="#fbbf24", linestyle="--", lw=1.5, alpha=0.8, label=f"I* ({chamber.equilibrium_current:.2f}A)")
    line_I, = ax_I.plot([], [], color="#f43f5e", lw=2.2, label="AI Current I(t)")
    dot_I, = ax_I.plot([], [], marker="o", color="#f43f5e", markersize=6)
    ax_I.set_ylabel("Current I (A)", color="#f43f5e", fontsize=9.5, fontweight="bold")
    ax_I.set_title("ELECTROMAGNET POWER SIGNAL", color="#e2e8f0", fontsize=10.5, fontweight="bold", pad=6)
    ax_I.legend(loc="upper right", fontsize=8, facecolor="#1f2937", framealpha=0.6)

    # Ax 3: Velocity v(t)
    ax_v.set_xlim(0, duration)
    ax_v.set_ylim(-3.5, 3.5)
    ax_v.axhline(0, color="#6b7280", linestyle="--", lw=1.2, alpha=0.8)
    line_v, = ax_v.plot([], [], color="#22c55e", lw=2.0, label="Velocity dy/dt")
    dot_v, = ax_v.plot([], [], marker="o", color="#22c55e", markersize=6)
    ax_v.set_ylabel("Velocity (m/s)", color="#22c55e", fontsize=9.5, fontweight="bold")
    ax_v.set_xlabel("Time (seconds)", color="#9ca3af", fontsize=9.5)
    ax_v.set_title("VELOCITY DAMPING (STABILIZATION)", color="#e2e8f0", fontsize=10.5, fontweight="bold", pad=6)
    ax_v.legend(loc="lower right", fontsize=8, facecolor="#1f2937", framealpha=0.6)

    fig.suptitle("✦ NEURAL CONTROLLER MAGNETIC LEVITATION SIMULATOR ✦", fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)

    # =============================================================
    # ANIMATION UPDATE FUNCTION
    # =============================================================
    def update(frame):
        idx = frame_indices[frame]
        t = sim_times[idx]
        y_curr = sim_y[idx]
        v_curr = sim_v[idx]
        I_curr = sim_I[idx]
        fmag_curr = sim_fmag[idx]
        fgrav = chamber.m * chamber.g

        # Update Ball Position
        ball_patch.center = (0, y_curr)
        shine_patch.center = (0.008, y_curr + 0.012)

        # Update Shadow: closer to floor => darker and wider
        shadow_dist = max(0.001, y_curr)
        shadow_alpha = np.clip(0.85 * np.exp(-shadow_dist / 0.15), 0.05, 0.85)
        shadow_width = np.clip(0.12 - 0.08 * (shadow_dist / chamber.H), 0.03, 0.12)
        shadow_patch.width = shadow_width
        shadow_patch.set_alpha(shadow_alpha)

        # Update Laser Beam
        laser_line.set_data([0, 0], [chamber.H, y_curr + ball_radius])
        laser_dot.set_data([0], [y_curr + ball_radius])

        # Update Dynamic Magnetic Flux Glow based on Current
        flux_intensity = np.clip(I_curr / chamber.max_current, 0.05, 1.0)
        flux_glow.width = 0.4 + 0.5 * flux_intensity
        flux_glow.height = 0.2 + 0.4 * flux_intensity
        flux_glow.set_alpha(0.08 + 0.35 * flux_intensity)

        # Update Force Arrows
        arrow_fmag.xy = (0, y_curr + np.clip(fmag_curr * 0.12, 0.04, 0.25))
        arrow_fmag.set_position((0, y_curr + ball_radius))

        arrow_fgrav.xy = (0, y_curr - np.clip(fgrav * 0.12, 0.04, 0.25))
        arrow_fgrav.set_position((0, y_curr - ball_radius))

        # Status mode calculation
        error = abs(y_curr - chamber.y_target)
        if error < 0.01 and abs(v_curr) < 0.04:
            status_text = "✦ HOVER LOCKED (STABLE)"
            status_color = "#34d399"
        elif v_curr < -0.1 and y_curr < chamber.y_target:
            status_text = "⚡ BRAKING UPWARD BOOST"
            status_color = "#f43f5e"
        elif y_curr > chamber.y_target + 0.05:
            status_text = "▼ GRAVITY RESTORE DESCENT"
            status_color = "#fbbf24"
        else:
            status_text = "✦ CATCHING OBJECT"
            status_color = "#38bdf8"

        hud_info = (
            f"STATE: {status_text}\n"
            f"TIME:     {t:.3f} s\n"
            f"ALTITUDE: {y_curr:.4f} m\n"
            f"VELOCITY: {v_curr:+.3f} m/s\n"
            f"CURRENT:  {I_curr:.3f} A\n"
            f"F_MAG:    {fmag_curr:.3f} N\n"
            f"F_GRAV:   {fgrav:.3f} N"
        )
        hud_text.set_text(hud_info)

        # Update Oscilloscopes
        t_hist = sim_times[:idx + 1]
        line_y.set_data(t_hist, sim_y[:idx + 1])
        dot_y.set_data([t], [y_curr])

        line_I.set_data(t_hist, sim_I[:idx + 1])
        dot_I.set_data([t], [I_curr])

        line_v.set_data(t_hist, sim_v[:idx + 1])
        dot_v.set_data([t], [v_curr])

        return (
            ball_patch, shine_patch, shadow_patch, laser_line, laser_dot,
            flux_glow, hud_text, line_y, dot_y, line_I, dot_I, line_v, dot_v
        )

    print(f"Generating animation with {n_frames} frames ({fps} FPS)...")
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    os.makedirs("assets", exist_ok=True)
    gif_path = os.path.join("assets", "antigravity_levitation.gif")

    if save_gif:
        print(f"Saving high-resolution animation to {gif_path} (this may take ~15 seconds)...")
        anim.save(gif_path, writer=animation.PillowWriter(fps=fps))
        print(f"[Animation Saved] Rendered GIF saved to: {gif_path}")
        if sys.platform.startswith("win"):
            try:
                os.startfile(os.path.abspath(gif_path))
                print(f"[Opened in Viewer] Launched {gif_path}")
            except Exception as e:
                print(f"Viewer launch notice: {e}")

    if show_window:
        print("[Displaying Interactive Animation Window] Opening live window...")
        plt.show()

    plt.close()
    return gif_path


if __name__ == "__main__":
    create_antigravity_animation(save_gif=True, show_window=True)
