"""
dataset.py - Synthetic Dataset Generator for the Antigravity Controller
Generates states [current_distance, current_velocity] and optimal target current [optimal_current]
using physics-based Feedback Linearization control across:
1. Broad State-Space Grid Exploration (dense coverage of the chamber).
2. Dynamic Trajectory Rollouts (realistic approach curves and transient states).
3. Realistic sensor noise injection for model robustness.
"""

import os
import numpy as np
from physics import MagneticLevitationChamber


def generate_synthetic_dataset(
    n_samples: int = 12000,
    chamber: MagneticLevitationChamber = None,
    noise_std_y: float = 0.003,
    noise_std_v: float = 0.015,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset of states [y, v] and optimal target current [I].
    """
    if chamber is None:
        chamber = MagneticLevitationChamber()

    rng = np.random.RandomState(random_seed)

    # 1. State-Space Random Exploration (70% of data)
    n_random = int(n_samples * 0.7)
    # Distance in [0.02, 0.98] meters
    y_rand = rng.uniform(0.02, chamber.H - 0.02, size=(n_random, 1))
    # Velocity in [-3.5, 3.5] m/s
    v_rand = rng.uniform(-3.5, 3.5, size=(n_random, 1))

    # Add extra dense sampling around hover point y_target +/- 0.1m and v=0
    n_hover = int(n_samples * 0.15)
    y_hover = rng.normal(loc=chamber.y_target, scale=0.06, size=(n_hover, 1))
    y_hover = np.clip(y_hover, 0.02, chamber.H - 0.02)
    v_hover = rng.normal(loc=0.0, scale=0.4, size=(n_hover, 1))

    # Add extra dense sampling near crash boundary y in [0.01, 0.15] (critical safety zone)
    n_crash_zone = int(n_samples * 0.15)
    y_crash = rng.uniform(0.01, 0.18, size=(n_crash_zone, 1))
    v_crash = rng.uniform(-4.0, 1.0, size=(n_crash_zone, 1))

    # Combine sampled states
    y_all = np.vstack([y_rand, y_hover, y_crash])
    v_all = np.vstack([v_rand, v_hover, v_crash])

    # Compute optimal current for every state
    currents = []
    for y_val, v_val in zip(y_all.ravel(), v_all.ravel()):
        i_opt = chamber.compute_optimal_control(y_val, v_val)
        currents.append(i_opt)

    currents = np.array(currents, dtype=np.float64).reshape(-1, 1)

    # 2. Dynamic Trajectory Rollouts to capture realistic continuous dynamics
    trajectory_states = []
    trajectory_currents = []

    drop_heights = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
    initial_velocities = [-2.0, -1.0, 0.0, 1.0, 2.0]

    for y_init in drop_heights:
        for v_init in initial_velocities:
            chamber.reset(initial_height=y_init, initial_velocity=v_init)
            for _ in range(150):  # 150 steps = 0.75 seconds of stabilization
                y_curr, v_curr = chamber.y, chamber.v
                i_cmd = chamber.compute_optimal_control(y_curr, v_curr)

                trajectory_states.append([y_curr, v_curr])
                trajectory_currents.append([i_cmd])

                chamber.step(i_cmd)

    traj_X = np.array(trajectory_states, dtype=np.float64)
    traj_y = np.array(trajectory_currents, dtype=np.float64)

    # Combine static samples and trajectory samples
    X_clean = np.vstack([np.hstack([y_all, v_all]), traj_X])
    y_clean = np.vstack([currents, traj_y])

    # 3. Inject realistic sensor noise into inputs
    X_noisy = X_clean.copy()
    X_noisy[:, 0] += rng.normal(0.0, noise_std_y, size=X_clean.shape[0])
    X_noisy[:, 1] += rng.normal(0.0, noise_std_v, size=X_clean.shape[0])
    # Distance cannot be negative
    X_noisy[:, 0] = np.clip(X_noisy[:, 0], 0.001, chamber.H)

    # Shuffle dataset
    perm = rng.permutation(len(X_noisy))
    X_final = X_noisy[perm]
    y_final = y_clean[perm]

    return X_final, y_final


def save_and_split_dataset(
    output_dir: str = "data",
    n_samples: int = 15000,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict:
    """
    Generate dataset, split into train/val/test, and save to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    chamber = MagneticLevitationChamber()
    print(f"Generating synthetic dataset with physics chamber (Target Height = {chamber.y_target}m)...")
    X, y = generate_synthetic_dataset(n_samples=n_samples, chamber=chamber)

    N = len(X)
    n_val = int(N * val_ratio)
    n_test = int(N * test_ratio)
    n_train = N - n_val - n_test

    train_X, train_y = X[:n_train], y[:n_train]
    val_X, val_y = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    test_X, test_y = X[n_train + n_val :], y[n_train + n_val :]

    save_path = os.path.join(output_dir, "antigravity_dataset.npz")
    np.savez_compressed(
        save_path,
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        test_X=test_X,
        test_y=test_y,
        equilibrium_current=chamber.equilibrium_current,
        target_height=chamber.y_target,
    )
    print(f"Dataset successfully created and saved to: {save_path}")
    print(f"Total samples: {N} | Train: {len(train_X)} | Val: {len(val_X)} | Test: {len(test_X)}")
    print(f"Equilibrium Current at {chamber.y_target}m: {chamber.equilibrium_current:.4f} A")

    return {
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_y": test_y,
    }


if __name__ == "__main__":
    save_and_split_dataset()
