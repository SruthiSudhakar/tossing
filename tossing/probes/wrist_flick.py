"""wrist_flick: Wrist flick probe.

Apply a torque impulse and measure angular velocity response.
Returns: angular_velocity_response, angular_deceleration, precession_detected.
Diagnostic purpose: moment of inertia (angular acceleration from known torque),
                    CoM offset (translational drift during spin).
"""

from __future__ import annotations

import numpy as np

from tossing.probes import ProbeController, register_probe
from tossing.types import ProbeResult


OBSERVE_TIME = 0.5  # fixed total observation window (seconds)


@register_probe("wrist_flick")
class WristFlickProbe(ProbeController):

    PARAM_SPEC = {
        "torque": (1.0, 0.1, 5.0),            # N*m applied for a brief impulse
        "impulse_duration": (0.05, 0.01, 0.2),  # seconds of torque application
    }

    def execute(self, env, params: dict | None = None) -> ProbeResult:
        p = self.resolve_params(params)
        torque_magnitude = p["torque"]
        impulse_duration = p["impulse_duration"]

        # Allow 3D rotation to detect precession
        env._enforce_2d = False

        env._release(
            linear_vel=np.array([0.0, 0.0, 1.5]),
            angular_vel=np.array([0.0, 0.0, 0.0]),
        )

        release_pos = env.object_pos.copy()
        t_start = env.sim_time
        trajectory = []
        omega_samples = []
        pos_samples = []
        steps_per_frame = max(1, int(1.0 / (30 * env.timestep)))
        impulse_steps = int(impulse_duration / env.timestep)
        total_steps = int(OBSERVE_TIME / env.timestep)

        for i in range(total_steps):
            if i < impulse_steps:
                env.data.xfrc_applied[env.object_body_id, 3:] = [0, torque_magnitude, 0]
            else:
                env.data.xfrc_applied[env.object_body_id, 3:] = [0, 0, 0]

            env._step()

            if i % steps_per_frame == 0:
                trajectory.append(env._get_object_state())
                vel = env._get_object_vel()
                omega_samples.append(vel[3:].copy())
                pos_samples.append(env.object_pos.copy())

        env._enforce_2d = True

        omega_samples = np.array(omega_samples)
        pos_samples = np.array(pos_samples)

        omega_y = omega_samples[:, 1] if len(omega_samples) > 0 else np.array([0.0])

        peak_idx = min(max(1, int(impulse_duration * 30)), len(omega_y) - 1)
        angular_velocity_response = float(np.max(np.abs(omega_y[:peak_idx + 2])))

        if len(omega_y) >= peak_idx + 2:
            angular_deceleration = float(
                (np.abs(omega_y[peak_idx]) - np.abs(omega_y[-1])) / (OBSERVE_TIME - impulse_duration)
            )
        else:
            angular_deceleration = 0.0

        if len(omega_samples) > 0:
            max_wx = float(np.max(np.abs(omega_samples[:, 0])))
            max_wz = float(np.max(np.abs(omega_samples[:, 2])))
            precession_detected = max_wx > 0.5 or max_wz > 0.5
        else:
            precession_detected = False

        if len(pos_samples) > 1:
            drift = np.linalg.norm(pos_samples[-1, :2] - pos_samples[0, :2])
        else:
            drift = 0.0

        observations = {
            "angular_velocity_response": angular_velocity_response,
            "angular_deceleration": angular_deceleration,
            "precession_detected": float(precession_detected),
            "translational_drift": float(drift),
        }

        traj_array = np.array(trajectory) if trajectory else np.zeros((0, 6))

        return ProbeResult(
            probe_type="wrist_flick",
            observations=observations,
            trajectory=traj_array,
            cost=1,
            params=p,
        )
