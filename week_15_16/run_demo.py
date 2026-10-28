"""
run_demo.py - Interactive Launcher & Master Demonstration Script
Launches all phases of the AI-Powered Antigravity Levitation Project:
1. Real-Time Deployment & Multi-Drop Catch Simulation (with graphs automatically opening!)
2. User Challenge Benchmark: Crash-Penalty Loss vs Standard MSE Loss (with comparison graphs!)
3. Real-Time Animated Antigravity Chamber GIF & Interactive Window
4. Interactive 60 FPS Web Simulator in the browser!
"""

import os
import sys
import webbrowser

# Ensure clean UTF-8 printing on Windows terminals
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulate import run_simulation
from challenge_experiment import run_challenge_benchmark
from animate import create_antigravity_animation


def main():
    print("=" * 75)
    print("[*] AI-POWERED 'ANTIGRAVITY' LEVITATION CONTROLLER DEMO")
    print("    Neural Network Controller for Magnetic Levitation (Antigravity Chamber)")
    print("=" * 75)
    print("Select an option:")
    print("  [1] Run Multi-Drop Catch Simulation (Opens stylish telemetry dashboard)")
    print("  [2] Run User Challenge Benchmark (Crash Penalty vs MSE Loss with graphs)")
    print("  [3] Generate & Play Chamber Animation (Futuristic glowing electromagnet GIF)")
    print("  [4] Launch Interactive Web Chamber Simulator in Browser (Mouse Drag & Drop!)")
    print("  [5] Run ALL Demos Sequentially (Full Showcase)")
    print("=" * 75)

    choice = sys.argv[1] if len(sys.argv) > 1 else "5"
    print(f"Selected Option: [{choice}]")

    if choice in ("1", "5"):
        print("\n>>> [1/4] Running Deployment Simulation & Opening Telemetry Graphs...")
        run_simulation(open_graph=True)

    if choice in ("2", "5"):
        print("\n>>> [2/4] Running User Challenge: Crash Penalty Benchmark...")
        run_challenge_benchmark(open_graph=True)

    if choice in ("3", "5"):
        print("\n>>> [3/4] Generating & Opening Chamber Animation...")
        gif_file = create_antigravity_animation(duration=2.0, fps=25, save_gif=True, show_window=False)
        print(f"Animation rendered to: {gif_file}")

    if choice in ("4", "5"):
        print("\n>>> [4/4] Launching Interactive Web Visualizer in Browser...")
        web_index = os.path.abspath(os.path.join("web", "index.html"))
        if os.path.exists(web_index):
            webbrowser.open(f"file:///{web_index}")
            print(f"Opened web simulator at: file:///{web_index}")
        else:
            print("Web files not found.")

    print("\n" + "=" * 75)
    print("[SUCCESS] DEMO SUITE COMPLETE! All graphs, models, and animations are active.")
    print("=" * 75)


if __name__ == "__main__":
    main()
