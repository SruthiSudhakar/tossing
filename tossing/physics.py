"""Physics utilities: drag force, ballistic math for validation."""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.225  # kg/m^3 at sea level


def apply_drag(data, body_id: int, c_d: float, area: float, rho: float = AIR_DENSITY):
    """Apply aerodynamic drag force to a MuJoCo body.

    F_drag = -0.5 * rho * c_d * A * |v| * v

    Applied via data.xfrc_applied (external force in world frame).
    """
    # cvel layout: [wx, wy, wz, vx, vy, vz]
    vel = data.cvel[body_id, 3:].copy()
    speed = np.linalg.norm(vel)
    if speed > 1e-8:
        f_drag = -0.5 * rho * c_d * area * speed * vel
        data.xfrc_applied[body_id, :3] = f_drag


def ballistic_apex(v_z: float) -> float:
    """Analytical apex height for vertical launch (no drag).

    h = v_z^2 / (2g)
    """
    return v_z ** 2 / (2 * GRAVITY)


def ballistic_hang_time(v_z: float) -> float:
    """Time for vertical toss to return to launch height (no drag).

    t = 2 * v_z / g
    """
    return 2 * v_z / GRAVITY


def ballistic_fall_time(h: float) -> float:
    """Time to fall height h from rest (no drag).

    t = sqrt(2h / g)
    """
    return np.sqrt(2 * h / GRAVITY)


def ballistic_range(v: float, theta_deg: float, h: float = 0.0) -> float:
    """Horizontal range for projectile launched at angle theta from height h (no drag).

    For h=0: R = v^2 * sin(2*theta) / g
    For h>0: R = (v*cos(theta)/g) * (v*sin(theta) + sqrt((v*sin(theta))^2 + 2*g*h))
    """
    theta = np.radians(theta_deg)
    vx = v * np.cos(theta)
    vz = v * np.sin(theta)
    if h <= 0:
        return v ** 2 * np.sin(2 * theta) / GRAVITY
    # General case with launch height
    discriminant = vz ** 2 + 2 * GRAVITY * h
    if discriminant < 0:
        return 0.0
    t_flight = (vz + np.sqrt(discriminant)) / GRAVITY
    return vx * t_flight


def ballistic_flight_time(v: float, theta_deg: float, h: float = 0.0) -> float:
    """Flight time for projectile launched at angle theta from height h (no drag)."""
    theta = np.radians(theta_deg)
    vz = v * np.sin(theta)
    discriminant = vz ** 2 + 2 * GRAVITY * h
    if discriminant < 0:
        return 0.0
    return (vz + np.sqrt(discriminant)) / GRAVITY
