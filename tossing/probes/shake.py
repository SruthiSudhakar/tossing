"""shake: Small shake probe.

Oscillate gripper vertically while object remains grasped. Tunable
amplitude, frequency, duration. Returns: peak_force, damping_ratio,
perceived_resistance.
Diagnostic purpose: inertia response, internal mass distribution.
"""

from __future__ import annotations

import numpy as np

from tossing.probes import ProbeController, register_probe
from tossing.types import ProbeResult


@register_probe("shake")
class ShakeProbe(ProbeController):

    PARAM_SPEC = {
        "frequency": (4.0, 0.5, 20.0),     # Hz
        "amplitude": (0.05, 0.01, 0.15),   # meters
        "duration": (1.0, 0.3, 2.0),       # seconds
    }

    def execute(self, env, params: dict | None = None) -> ProbeResult:
        p = self.resolve_params(params)
        shake_freq = p["frequency"]
        shake_amplitude = p["amplitude"]
        shake_duration = p["duration"]

        trajectory = []
        forces = []
        steps_per_frame = max(1, int(1.0 / (30 * env.timestep)))
        n_steps = int(shake_duration / env.timestep)

        for i in range(n_steps):
            t = i * env.timestep
            target_z = shake_amplitude * np.sin(2 * np.pi * shake_freq * t)

            kp = 500.0
            grip_qpos_adr = env.model.jnt_qposadr[
                env.model.joint('gripper_z').id
            ]
            current_z = env.data.qpos[grip_qpos_adr]
            env.data.ctrl[env.gripper_act_id] = kp * (target_z - current_z)

            env._step()

            if i % steps_per_frame == 0:
                trajectory.append(env._get_object_state())
                obj_acc = env.data.qacc[env.obj_qvel_adr:env.obj_qvel_adr + 3]
                force_magnitude = float(np.linalg.norm(obj_acc) * env.obj.mass)
                forces.append(force_magnitude)

        env.data.ctrl[:] = 0

        forces = np.array(forces)
        peak_force = float(np.max(forces)) if len(forces) > 0 else 0.0

        omega = 2 * np.pi * shake_freq
        expected_peak = env.obj.mass * shake_amplitude * omega ** 2
        perceived_resistance = peak_force / expected_peak if expected_peak > 1e-8 else 0.0

        if len(forces) > 4:
            damping_ratio = float(np.mean(forces) / (peak_force + 1e-8))
        else:
            damping_ratio = 0.0

        observations = {
            "peak_force": peak_force,
            "damping_ratio": damping_ratio,
            "perceived_resistance": perceived_resistance,
        }

        traj_array = np.array(trajectory) if trajectory else np.zeros((0, 6))

        return ProbeResult(
            probe_type="shake",
            observations=observations,
            trajectory=traj_array,
            cost=1,
            params=p,
        )
