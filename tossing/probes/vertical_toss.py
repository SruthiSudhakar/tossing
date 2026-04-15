"""vertical_toss: Vertical micro-toss probe.

Launch object straight up at 2 m/s.
Returns: apex_height, hang_time, landing_offset.
Diagnostic purpose: mass (hang time), drag (apex delta from ballistic prediction).
"""

from __future__ import annotations

import numpy as np

from tossing.probes import ProbeController, register_probe
from tossing.types import ProbeResult


LAUNCH_SPEED = 2.0  # m/s upward


@register_probe("vertical_toss")
class VerticalTossProbe(ProbeController):

    def execute(self, env) -> ProbeResult:
        release_pos = env.object_pos.copy()
        release_z = release_pos[2]
        release_x = release_pos[0]

        # Release straight up
        env._release(linear_vel=np.array([0.0, 0.0, LAUNCH_SPEED]))

        # Track trajectory until object returns below release height or hits ground
        max_z = release_z
        trajectory = []
        steps_per_frame = max(1, int(1.0 / (30 * env.timestep)))
        t_start = env.sim_time

        for i in range(10000):  # max ~20s
            env._step()
            pos = env.object_pos
            z = pos[2]

            if i % steps_per_frame == 0:
                state = env._get_object_state()
                trajectory.append(state)

            if z > max_z:
                max_z = z

            # Stop when object returns to release height (descending) or hits ground
            vel_z = env._get_object_vel()[2]
            if z <= release_z and vel_z < 0 and i > 10:
                break
            if env._object_on_ground():
                break

        t_end = env.sim_time
        final_pos = env.object_pos

        apex_height = max_z - release_z
        hang_time = t_end - t_start
        landing_offset = final_pos[0] - release_x

        observations = {
            "apex_height": float(apex_height),
            "hang_time": float(hang_time),
            "landing_offset": float(landing_offset),
        }

        traj_array = np.array(trajectory) if trajectory else np.zeros((0, 6))

        return ProbeResult(
            probe_type="vertical_toss",
            observations=observations,
            trajectory=traj_array,
            cost=1,
        )
