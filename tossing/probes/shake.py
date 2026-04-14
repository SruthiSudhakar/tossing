"""P5: Small shake probe.

Oscillate gripper ±5cm at 4 Hz for 1 second while object remains grasped.
Returns: peak_force, damping_ratio, perceived_resistance.
Diagnostic purpose: inertia response, internal mass distribution.
"""

from __future__ import annotations

import numpy as np

from tossing.probes import ProbeController, register_probe
from tossing.types import ProbeResult


SHAKE_FREQ = 4.0  # Hz
SHAKE_AMPLITUDE = 0.05  # meters
SHAKE_DURATION = 1.0  # seconds


@register_probe("P5")
class ShakeProbe(ProbeController):

    def execute(self, env) -> ProbeResult:
        # Object stays grasped — oscillate the gripper joint
        trajectory = []
        forces = []
        steps_per_frame = max(1, int(1.0 / (30 * env.timestep)))
        n_steps = int(SHAKE_DURATION / env.timestep)

        t_start = env.sim_time

        for i in range(n_steps):
            t = i * env.timestep
            # Sinusoidal position command for gripper z-joint
            target_z = SHAKE_AMPLITUDE * np.sin(2 * np.pi * SHAKE_FREQ * t)

            # PD control of gripper_z joint
            grip_jnt_id = env.model.jnt_qposadr[
                env.model.jnt_dofadr.__class__  # just use the actuator directly
            ] if False else None

            # Use actuator to drive gripper
            # The gripper_motor actuator controls gripper_z joint
            # Use a simple proportional controller
            kp = 500.0
            grip_qpos_adr = env.model.jnt_qposadr[
                env.model.joint('gripper_z').id
            ]
            current_z = env.data.qpos[grip_qpos_adr]
            env.data.ctrl[env.gripper_act_id] = kp * (target_z - current_z)

            env._step()

            if i % steps_per_frame == 0:
                trajectory.append(env._get_object_state())

                # Measure the constraint force magnitude from the weld
                # xfrc_applied reflects external forces; for constraint forces,
                # we look at the object's acceleration vs expected
                obj_acc = env.data.qacc[env.obj_qvel_adr:env.obj_qvel_adr + 3]
                force_magnitude = float(np.linalg.norm(obj_acc) * env.obj.mass)
                forces.append(force_magnitude)

        # Reset controls
        env.data.ctrl[:] = 0

        forces = np.array(forces)
        peak_force = float(np.max(forces)) if len(forces) > 0 else 0.0

        # Perceived resistance: peak force / (mass * amplitude * omega^2)
        # This normalizes by the expected force for a rigid body
        omega = 2 * np.pi * SHAKE_FREQ
        expected_peak = env.obj.mass * SHAKE_AMPLITUDE * omega ** 2
        perceived_resistance = peak_force / expected_peak if expected_peak > 1e-8 else 0.0

        # Damping ratio: estimate from force waveform (simplified)
        # Higher damping = more phase lag between command and response
        if len(forces) > 4:
            # Use ratio of mean to peak as a proxy for damping
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
            probe_type="P5",
            observations=observations,
            trajectory=traj_array,
            cost=1,
        )
