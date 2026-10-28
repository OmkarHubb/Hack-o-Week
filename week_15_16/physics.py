"""
physics.py - Magnetic Levitation Physics Environment (The "Antigravity" Chamber)
Simulates a metallic sphere suspended in mid-air under the influence of:
1. Earth's gravitational acceleration (pulling downwards at g = 9.81 m/s^2).
2. Electromagnetic attraction from the ceiling (pulling upwards, inverse-square law).
3. Aerodynamic drag (air resistance damping velocity).
4. Boundary constraints (floor at y=0, electromagnet ceiling at y=H).
"""

import numpy as np


class MagneticLevitationChamber:
    """
    Physical simulation of an Antigravity Magnetic Levitation Chamber.

    Coordinate System:
    - y = 0.0 m is the chamber floor (CRASH boundary).
    - y = H (e.g. 1.0 m) is the chamber ceiling where the electromagnet is located.
    - y_target (e.g. 0.5 m) is the target hover height.
    - Distance to electromagnet: d = H - y + epsilon (where epsilon prevents division by zero).
    - Velocity: v = dy/dt (positive = moving upwards towards magnet, negative = falling).
    """

    def __init__(
        self,
        chamber_height: float = 1.0,
        target_height: float = 0.5,
        mass: float = 0.05,            # 50 grams
        gravity: float = 9.81,          # m/s^2
        mag_constant: float = 0.030656, # N * m^2 / A^2 (chosen so I* ~ 2.0A at y=0.5m)
        drag_coeff: float = 0.015,      # N * s / m (air damping)
        coil_offset: float = 0.02,      # Coil physical casing offset (meters)
        max_current: float = 10.0,      # Max electromagnet current in Amperes
        dt: float = 0.005,              # Simulation time step in seconds (200 Hz)
        restitution: float = 0.1,       # Floor bounce coefficient of restitution
    ):
        self.H = chamber_height
        self.y_target = target_height
        self.m = mass
        self.g = gravity
        self.k_mag = mag_constant
        self.c_drag = drag_coeff
        self.eps = coil_offset
        self.max_current = max_current
        self.dt = dt
        self.restitution = restitution

        # Internal state: [y, v]
        self.y = target_height
        self.v = 0.0
        self.time = 0.0
        self.crashed = False
        self.hit_ceiling = False

    @property
    def equilibrium_current(self) -> float:
        """
        Analytically calculate the steady-state electrical current needed to
        perfectly hover the object at the target height (v=0, a=0).
        F_mag = F_grav => k_mag * I^2 / (H - y_target + eps)^2 = m * g
        => I* = (H - y_target + eps) * sqrt(m * g / k_mag)
        """
        dist_to_magnet = self.H - self.y_target + self.eps
        return dist_to_magnet * np.sqrt((self.m * self.g) / self.k_mag)

    def magnetic_force(self, y: float, current: float) -> float:
        """
        Upward magnetic force pulling object towards the top electromagnet.
        F_mag = k_mag * I^2 / (H - y + eps)^2
        """
        current = np.clip(current, 0.0, self.max_current)
        dist = np.maximum(self.eps, self.H - y + self.eps)
        return self.k_mag * (current ** 2) / (dist ** 2)

    def compute_acceleration(self, y: float, v: float, current: float) -> float:
        """
        Net acceleration a = F_net / m:
        a = (F_mag - F_gravity - F_drag) / m
        """
        f_mag = self.magnetic_force(y, current)
        f_grav = self.m * self.g
        f_drag = self.c_drag * v
        f_net = f_mag - f_grav - f_drag
        return f_net / self.m

    def reset(self, initial_height: float = None, initial_velocity: float = 0.0) -> np.ndarray:
        """
        Reset chamber to initial state.
        """
        if initial_height is None:
            self.y = self.y_target
        else:
            self.y = np.clip(initial_height, 0.0, self.H)
        self.v = initial_velocity
        self.time = 0.0
        self.crashed = False
        self.hit_ceiling = False
        return np.array([self.y, self.v], dtype=np.float64)

    def step(self, current: float) -> tuple[np.ndarray, float, bool, dict]:
        """
        Advance physics simulation by dt using 4th-order Runge-Kutta (RK4) integration.
        Returns: (next_state, reward/error, done, info)
        """
        current = float(np.clip(current, 0.0, self.max_current))
        y0 = self.y
        v0 = self.v
        dt = self.dt

        # RK4 Integration for superior numerical accuracy
        k1_v = self.compute_acceleration(y0, v0, current)
        k1_y = v0

        k2_v = self.compute_acceleration(y0 + 0.5 * dt * k1_y, v0 + 0.5 * dt * k1_v, current)
        k2_y = v0 + 0.5 * dt * k1_v

        k3_v = self.compute_acceleration(y0 + 0.5 * dt * k2_y, v0 + 0.5 * dt * k2_v, current)
        k3_y = v0 + 0.5 * dt * k2_v

        k4_v = self.compute_acceleration(y0 + dt * k3_y, v0 + dt * k3_v, current)
        k4_y = v0 + dt * k3_v

        y_next = y0 + (dt / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
        v_next = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

        # Boundary checks
        crashed = False
        hit_ceiling = False

        if y_next <= 0.0:
            # Crash on chamber floor
            y_next = 0.0
            if v_next < 0:
                v_next = -v_next * self.restitution
                if abs(v_next) < 0.05:
                    v_next = 0.0
            crashed = True
        elif y_next >= self.H:
            # Contact with ceiling electromagnet
            y_next = self.H
            if v_next > 0:
                v_next = -v_next * self.restitution
                if abs(v_next) < 0.05:
                    v_next = 0.0
            hit_ceiling = True

        self.y = y_next
        self.v = v_next
        self.time += dt
        self.crashed = crashed
        self.hit_ceiling = hit_ceiling

        error = abs(self.y - self.y_target)
        done = False
        info = {
            "time": self.time,
            "y": self.y,
            "v": self.v,
            "current": current,
            "crashed": crashed,
            "hit_ceiling": hit_ceiling,
            "f_mag": self.magnetic_force(self.y, current),
            "f_grav": self.m * self.g,
        }
        return np.array([self.y, self.v], dtype=np.float64), error, done, info

    def compute_optimal_control(
        self,
        y: float,
        v: float,
        omega_n: float = 12.0,  # Natural frequency (controller response speed)
        zeta: float = 1.0,      # Damping ratio (1.0 = critically damped, zero overshoot)
    ) -> float:
        """
        Control-theoretic ground truth using Feedback Linearization.
        Specifies second-order error dynamics:
            e = y - y_target
            e'' + 2 * zeta * omega_n * e' + omega_n^2 * e = 0
            => a_desired = -2 * zeta * omega_n * v - omega_n^2 * (y - y_target)

        Since a = (F_mag - F_grav - F_drag) / m, we solve for required current:
            F_mag_needed = m * (a_desired + g) + c_drag * v
            I_needed = dist * sqrt(max(0, F_mag_needed) / k_mag)
        """
        e = y - self.y_target
        a_desired = -2.0 * zeta * omega_n * v - (omega_n ** 2) * e

        # Required upward magnetic force
        f_mag_needed = self.m * (a_desired + self.g) + self.c_drag * v

        if f_mag_needed <= 0.0:
            return 0.0

        dist = max(self.eps, self.H - y + self.eps)
        i_needed = dist * np.sqrt(f_mag_needed / self.k_mag)
        return float(np.clip(i_needed, 0.0, self.max_current))
