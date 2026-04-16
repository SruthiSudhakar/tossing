"""forward_toss: Short forward toss probe.

Launch at `launch_angle` degrees at `launch_speed` m/s (defaults: 45, 1).
Returns: landing_distance, flight_time, lateral_drift.
Diagnostic purpose: drag (range shortfall), CoM offset (lateral drift).

The launch_speed cap is intentionally low so the probe cannot reach the
basket (max ballistic range ~0.86 m at h=1.5 m release height even with
the optimal angle), preventing the VLM from using the probe as a free
calibration throw.
"""

from __future__ import annotations

import numpy as np

from tossing.probes import ProbeController, register_probe
from tossing.types import ProbeResult


@register_probe("forward_toss")
class ForwardTossProbe(ProbeController):

    PARAM_SPEC = {
        "launch_speed": (1.0, 0.5, 1.5),
        "launch_angle": (45.0, 10.0, 80.0),
    }

    def execute(self, env, params: dict | None = None) -> ProbeResult:
        p = self.resolve_params(params)
        launch_speed = p["launch_speed"]
        launch_angle = p["launch_angle"]

        release_pos = env.object_pos.copy()
        theta = np.radians(launch_angle)
        vx = launch_speed * np.cos(theta)
        vz = launch_speed * np.sin(theta)

        env._release(linear_vel=np.array([vx, 0.0, vz]))

        t_start = env.sim_time
        trajectory = []
        steps_per_frame = max(1, int(1.0 / (30 * env.timestep)))

        for i in range(10000):
            env._step()

            if i % steps_per_frame == 0:
                trajectory.append(env._get_object_state())

            if env._object_on_ground() and i > 10:
                break

        t_end = env.sim_time
        final_pos = env.object_pos

        landing_distance = final_pos[0] - release_pos[0]
        flight_time = t_end - t_start
        from tossing.physics import ballistic_range
        h = release_pos[2]
        expected_range = ballistic_range(launch_speed, launch_angle, h)
        lateral_drift = landing_distance - expected_range

        observations = {
            "landing_distance": float(landing_distance),
            "flight_time": float(flight_time),
            "lateral_drift": float(lateral_drift),
        }

        traj_array = np.array(trajectory) if trajectory else np.zeros((0, 6))

        return ProbeResult(
            probe_type="forward_toss",
            observations=observations,
            trajectory=traj_array,
            cost=1,
            params=p,
        )
