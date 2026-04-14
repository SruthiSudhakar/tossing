"""P2: Short forward toss probe.

Launch at 45 degrees at 3 m/s.
Returns: landing_distance, flight_time, lateral_drift.
Diagnostic purpose: drag (range shortfall), CoM offset (lateral drift).
"""

from __future__ import annotations

import numpy as np

from tossing.probes import ProbeController, register_probe
from tossing.types import ProbeResult


LAUNCH_SPEED = 3.0  # m/s
LAUNCH_ANGLE = 45.0  # degrees


@register_probe("P2")
class ForwardTossProbe(ProbeController):

    def execute(self, env) -> ProbeResult:
        release_pos = env.object_pos.copy()
        theta = np.radians(LAUNCH_ANGLE)
        vx = LAUNCH_SPEED * np.cos(theta)
        vz = LAUNCH_SPEED * np.sin(theta)

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
        # In 2D, "lateral drift" is deviation from the expected zero-drag landing point
        # For a true 2D sim, y should be ~0, so we measure x-deviation from ballistic prediction
        from tossing.physics import ballistic_range
        h = release_pos[2]  # launch height above ground
        expected_range = ballistic_range(LAUNCH_SPEED, LAUNCH_ANGLE, h)
        lateral_drift = landing_distance - expected_range

        observations = {
            "landing_distance": float(landing_distance),
            "flight_time": float(flight_time),
            "lateral_drift": float(lateral_drift),
        }

        traj_array = np.array(trajectory) if trajectory else np.zeros((0, 6))

        return ProbeResult(
            probe_type="P2",
            observations=observations,
            trajectory=traj_array,
            cost=1,
        )
