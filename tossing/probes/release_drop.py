"""release_drop: Gentle release-drop probe.

Open gripper, let object fall from rest.
Returns: fall_time, coefficient_of_restitution, bounce_count.
Diagnostic purpose: mass (fall time if drag present), restitution.
"""

from __future__ import annotations

import numpy as np

from tossing.probes import ProbeController, register_probe
from tossing.types import ProbeResult


@register_probe("release_drop")
class ReleaseDropProbe(ProbeController):

    PARAM_SPEC = {
        "observe_time": (0.5, 0.1, 3.0),  # seconds after first contact to watch bounces
    }

    def execute(self, env, params: dict | None = None) -> ProbeResult:
        p = self.resolve_params(params)
        extra_observe_time = p["observe_time"]

        release_pos = env.object_pos.copy()
        drop_height = release_pos[2]

        env._release(linear_vel=np.array([0.0, 0.0, 0.0]))

        t_start = env.sim_time
        trajectory = []
        steps_per_frame = max(1, int(1.0 / (30 * env.timestep)))

        first_contact_time = None
        bounce_heights = []
        prev_z = drop_height
        going_up = False
        local_max_z = 0.0

        for i in range(10000):
            env._step()
            pos = env.object_pos
            z = pos[2]

            if i % steps_per_frame == 0:
                trajectory.append(env._get_object_state())

            if first_contact_time is None and env._object_on_ground():
                first_contact_time = env.sim_time

            if first_contact_time is not None:
                if z > prev_z and not going_up:
                    going_up = True
                    local_max_z = z
                elif going_up and z > local_max_z:
                    local_max_z = z
                elif going_up and z < prev_z:
                    if local_max_z > 0.02:
                        bounce_heights.append(local_max_z)
                    going_up = False

                if env.sim_time - first_contact_time > extra_observe_time:
                    break

            prev_z = z

        fall_time = (first_contact_time - t_start) if first_contact_time else (env.sim_time - t_start)

        if bounce_heights:
            cor = float(np.sqrt(bounce_heights[0] / drop_height))
        else:
            cor = 0.0

        observations = {
            "fall_time": float(fall_time),
            "coefficient_of_restitution": cor,
            "bounce_count": len(bounce_heights),
        }

        traj_array = np.array(trajectory) if trajectory else np.zeros((0, 6))

        return ProbeResult(
            probe_type="release_drop",
            observations=observations,
            trajectory=traj_array,
            cost=1,
            params=p,
        )
