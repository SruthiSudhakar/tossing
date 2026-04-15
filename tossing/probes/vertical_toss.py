"""vertical_toss: Vertical micro-toss probe.

Launch object straight up at `launch_speed` (default 2 m/s).
Returns: apex_height, hang_time, landing_offset.
Diagnostic purpose: mass (hang time), drag (apex delta from ballistic prediction).
"""

from __future__ import annotations

import numpy as np

from tossing.probes import ProbeController, register_probe
from tossing.types import ProbeResult


@register_probe("vertical_toss")
class VerticalTossProbe(ProbeController):

    PARAM_SPEC = {
        # name: (default, lo, hi)
        "launch_speed": (2.0, 0.5, 5.0),
    }

    def execute(self, env, params: dict | None = None) -> ProbeResult:
        p = self.resolve_params(params)
        launch_speed = p["launch_speed"]

        release_pos = env.object_pos.copy()
        release_z = release_pos[2]
        release_x = release_pos[0]

        env._release(linear_vel=np.array([0.0, 0.0, launch_speed]))

        max_z = release_z
        trajectory = []
        steps_per_frame = max(1, int(1.0 / (30 * env.timestep)))
        t_start = env.sim_time

        for i in range(10000):
            env._step()
            pos = env.object_pos
            z = pos[2]

            if i % steps_per_frame == 0:
                trajectory.append(env._get_object_state())

            if z > max_z:
                max_z = z

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
            params=p,
        )
