"""TossEnv — MuJoCo 2D tossing environment with parameterized objects."""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from tossing.types import ObjectSpec, ProbeResult, ThrowParams, ThrowResult
from tossing.physics import apply_drag


# Sampling rate for trajectory recording (Hz)
TRAJECTORY_HZ = 30

# Launcher height (fixed)
LAUNCHER_HEIGHT = 1.5  # meters

# Basket dimensions
BASKET_RADIUS = 0.15  # meters
BASKET_HEIGHT = 0.20  # meters
BASKET_FLOOR_HEIGHT = 0.3  # meters (height of basket bottom above ground)

# Ground contact threshold
GROUND_THRESHOLD = 0.05  # meters


def _build_mjcf(obj: ObjectSpec, basket_distance: float) -> str:
    """Build MJCF XML string for the tossing scene."""

    # Object geom specification
    if obj.geom_type == "sphere":
        geom_str = f'type="sphere" size="{obj.geom_size[0]}"'
    elif obj.geom_type == "box":
        sx, sy, sz = obj.geom_size[:3]
        geom_str = f'type="box" size="{sx} {sy} {sz}"'
    elif obj.geom_type == "cylinder":
        r, h = obj.geom_size[:2]
        geom_str = f'type="cylinder" size="{r} {h}"'
    elif obj.geom_type == "capsule":
        r, h = obj.geom_size[:2]
        geom_str = f'type="capsule" size="{r} {h}"'
    else:
        raise ValueError(f"Unknown geom_type: {obj.geom_type}")

    rgba_str = " ".join(f"{c:.3f}" for c in obj.rgba)
    com_x, com_y, com_z = obj.com_offset
    I = obj.inertia

    xml = f"""
    <mujoco model="toss2d">
      <option timestep="0.002" gravity="0 0 -9.81" integrator="Euler"/>

      <visual>
        <global offwidth="640" offheight="480"/>
        <rgba fog="0.95 0.95 1.0 1"/>
        <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0.2 0.2 0.2"/>
      </visual>

      <asset>
        <texture name="sky_tex" type="skybox" builtin="gradient"
                 rgb1="0.6 0.75 0.95" rgb2="0.95 0.95 1.0"
                 width="512" height="512"/>
        <texture name="ground_tex" type="2d" builtin="checker"
                 rgb1="0.92 0.90 0.85" rgb2="0.98 0.96 0.92"
                 width="256" height="256"/>
        <material name="ground_mat" texture="ground_tex"
                  texrepeat="4 4" reflectance="0.1"/>
      </asset>

      <default>
        <geom contype="1" conaffinity="1" condim="3"
              friction="0.8 0.005 0.001" solref="0.01 1.0"/>
      </default>

      <worldbody>
        <!-- Light -->
        <light pos="1 -2 3" dir="0 0.5 -1" diffuse="0.8 0.8 0.8"
               specular="0.3 0.3 0.3" castshadow="true"/>
        <light pos="-1 -1 2" dir="0.3 0.3 -0.5" diffuse="0.4 0.4 0.5"
               specular="0.1 0.1 0.1" castshadow="false"/>

        <!-- Ground plane -->
        <geom name="ground" type="plane" size="5 5 0.1"
              material="ground_mat" contype="1" conaffinity="1"/>

        <!-- Launcher on horizontal rail -->
        <body name="launcher" pos="0 0 {LAUNCHER_HEIGHT}">
          <joint name="rail_x" type="slide" axis="1 0 0"
                 range="-1 1" damping="5"/>
          <geom name="launcher_base" type="box" size="0.04 0.04 0.02"
                rgba="0.45 0.45 0.50 1" mass="1.0" contype="0" conaffinity="0"/>

          <!-- Gripper with vertical motion for shake probe -->
          <body name="gripper" pos="0 0 0">
            <joint name="gripper_z" type="slide" axis="0 0 1"
                   range="-0.1 0.1" damping="2"/>
            <geom name="gripper_geom" type="sphere" size="0.015"
                  rgba="0.55 0.55 0.60 1" mass="0.1" contype="0" conaffinity="0"/>
          </body>
        </body>

        <!-- Object -->
        <body name="object" pos="0 0 {LAUNCHER_HEIGHT}">
          <freejoint name="object_jnt"/>
          <inertial pos="{com_x} {com_y} {com_z}"
                    mass="{obj.mass}"
                    diaginertia="{I} {I} {I}"/>
          <geom name="obj_geom" {geom_str}
                rgba="{rgba_str}" mass="{obj.mass}"
                contype="1" conaffinity="1"/>
        </body>

        <!-- Basket -->
        <body name="basket" pos="{basket_distance} 0 {BASKET_FLOOR_HEIGHT}">
          <!-- Bottom -->
          <geom name="basket_bottom" type="cylinder"
                size="{BASKET_RADIUS} 0.01" pos="0 0 0"
                rgba="0.72 0.45 0.20 1" contype="1" conaffinity="1"/>
          <!-- Walls: 4 thin boxes arranged as a square approximation -->
          <geom name="bwall_px" type="box"
                size="0.01 {BASKET_RADIUS} {BASKET_HEIGHT}"
                pos="{BASKET_RADIUS} 0 {BASKET_HEIGHT}"
                rgba="0.72 0.45 0.20 0.85" contype="1" conaffinity="1"/>
          <geom name="bwall_nx" type="box"
                size="0.01 {BASKET_RADIUS} {BASKET_HEIGHT}"
                pos="-{BASKET_RADIUS} 0 {BASKET_HEIGHT}"
                rgba="0.72 0.45 0.20 0.85" contype="1" conaffinity="1"/>
          <geom name="bwall_py" type="box"
                size="{BASKET_RADIUS} 0.01 {BASKET_HEIGHT}"
                pos="0 {BASKET_RADIUS} {BASKET_HEIGHT}"
                rgba="0.72 0.45 0.20 0.85" contype="1" conaffinity="1"/>
          <geom name="bwall_ny" type="box"
                size="{BASKET_RADIUS} 0.01 {BASKET_HEIGHT}"
                pos="0 -{BASKET_RADIUS} {BASKET_HEIGHT}"
                rgba="0.72 0.45 0.20 0.85" contype="1" conaffinity="1"/>
        </body>

        <!-- Cameras -->
      </worldbody>

      <!-- Weld equality: grasps object to gripper -->
      <equality>
        <weld name="grasp" body1="gripper" body2="object"
              relpose="0 0 0 1 0 0 0" solref="0.005 1"/>
      </equality>

      <actuator>
        <motor name="rail_motor" joint="rail_x"
               ctrlrange="-100 100" gear="1"/>
        <motor name="gripper_motor" joint="gripper_z"
               ctrlrange="-50 50" gear="1"/>
      </actuator>

      <!-- Cameras defined outside worldbody for global positioning -->
    </mujoco>
    """
    return xml


class TossEnv:
    """2D sagittal-plane tossing environment.

    Usage:
        env = TossEnv(object_spec, basket_distance=2.0)
        result = env.run_probe("vertical_toss")  # run a diagnostic probe
        result = env.throw(ThrowParams(theta=45, v=5.0))  # attempt a throw
        images = env.render()             # get side+top view images
        env.reset(new_spec, new_distance) # swap object
    """

    def __init__(self, obj: ObjectSpec, basket_distance: float = 2.0):
        self.obj = obj
        self.basket_distance = basket_distance
        self.timestep = 0.002  # must match MJCF

        self._build(obj, basket_distance)

    def _build(self, obj: ObjectSpec, basket_distance: float):
        """Build MuJoCo model and data from object spec."""
        xml = _build_mjcf(obj, basket_distance)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Cache body/joint/actuator IDs
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.gripper_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.basket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "basket")

        # Freejoint qpos indices: [x, y, z, qw, qx, qy, qz]
        self.obj_jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_jnt")
        self.obj_qpos_adr = self.model.jnt_qposadr[self.obj_jnt_id]
        self.obj_qvel_adr = self.model.jnt_dofadr[self.obj_jnt_id]

        # Weld equality constraint ID
        self.grasp_eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")

        # Actuator IDs
        self.rail_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rail_motor")
        self.gripper_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_motor")

        # Renderer (lazy init)
        self._renderer = None

        # Video recording state (opt-in via start_recording).
        self._recording = False
        self._record_frames: list[np.ndarray] = []
        self._record_step_counter = 0
        self._record_steps_per_capture = 1

        # Initialize grasped state
        self._grasped = True
        self._enforce_2d = True
        self.data.eq_active[self.grasp_eq_id] = 1

        # Step forward to stabilize
        mujoco.mj_forward(self.model, self.data)

    def reset(self, obj: ObjectSpec | None = None, basket_distance: float | None = None):
        """Reset with a new object and/or basket distance."""
        if obj is not None:
            self.obj = obj
        if basket_distance is not None:
            self.basket_distance = basket_distance
        self._renderer = None
        self._build(self.obj, self.basket_distance)

    def _soft_reset(self):
        """Reset object back to gripper without rebuilding the model."""
        # Re-enable grasp
        self.data.eq_active[self.grasp_eq_id] = 1
        self._grasped = True

        # Reset object position to gripper position
        gripper_pos = self.data.xpos[self.gripper_body_id].copy()
        qa = self.obj_qpos_adr
        self.data.qpos[qa:qa + 3] = gripper_pos
        self.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]  # identity quaternion

        # Zero object velocities
        va = self.obj_qvel_adr
        self.data.qvel[va:va + 6] = 0

        # Zero applied forces
        self.data.xfrc_applied[self.object_body_id] = 0

        # Reset actuator controls
        self.data.ctrl[:] = 0

        # Reset launcher to center
        rail_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rail_x")
        grip_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_z")
        self.data.qpos[self.model.jnt_qposadr[rail_jnt]] = 0
        self.data.qpos[self.model.jnt_qposadr[grip_jnt]] = 0
        self.data.qvel[self.model.jnt_dofadr[rail_jnt]] = 0
        self.data.qvel[self.model.jnt_dofadr[grip_jnt]] = 0

        # Set object back to initial launcher position
        qa = self.obj_qpos_adr
        self.data.qpos[qa:qa + 3] = [0.0, 0.0, LAUNCHER_HEIGHT]
        self.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(self.model, self.data)

    def _release(self, linear_vel: np.ndarray | None = None,
                 angular_vel: np.ndarray | None = None):
        """Release the object from the gripper with optional initial velocity."""
        self.data.eq_active[self.grasp_eq_id] = 0
        self._grasped = False

        va = self.obj_qvel_adr
        if linear_vel is not None:
            self.data.qvel[va:va + 3] = linear_vel
        if angular_vel is not None:
            self.data.qvel[va + 3:va + 6] = angular_vel

    def _step(self):
        """Advance simulation by one timestep with drag and 2D enforcement."""
        # Apply drag force
        if not self._grasped:
            apply_drag(self.data, self.object_body_id,
                       self.obj.drag_coeff, self.obj.cross_section_area)

        mujoco.mj_step(self.model, self.data)

        # Enforce 2D: zero out y-translation and x/z-rotation
        if not self._grasped and self._enforce_2d:
            va = self.obj_qvel_adr
            self.data.qvel[va + 1] = 0  # vy = 0
            self.data.qvel[va + 3] = 0  # wx = 0
            self.data.qvel[va + 5] = 0  # wz = 0

            # Also zero y position drift
            qa = self.obj_qpos_adr
            self.data.qpos[qa + 1] = 0  # y = 0

        if self._recording:
            if self._record_step_counter % self._record_steps_per_capture == 0:
                self._record_frames.append(self.render_pair_array())
            self._record_step_counter += 1

    def _step_n(self, n: int):
        """Advance simulation by n timesteps."""
        for _ in range(n):
            self._step()

    def _get_object_pos(self) -> np.ndarray:
        """Get object center-of-mass position [x, y, z]."""
        return self.data.xpos[self.object_body_id].copy()

    def _get_object_vel(self) -> np.ndarray:
        """Get object velocity [vx, vy, vz, wx, wy, wz]."""
        va = self.obj_qvel_adr
        return self.data.qvel[va:va + 6].copy()

    def _get_object_state(self) -> np.ndarray:
        """Get [x, y, z, vx, vy, vz] for trajectory recording."""
        pos = self._get_object_pos()
        vel = self._get_object_vel()[:3]
        return np.concatenate([pos, vel])

    def _object_on_ground(self) -> bool:
        """Check if the object has hit the ground."""
        pos = self._get_object_pos()
        return pos[2] < GROUND_THRESHOLD

    def _object_settled(self, threshold: float = 0.01) -> bool:
        """Check if the object has settled (low velocity near ground)."""
        vel = self._get_object_vel()
        speed = np.linalg.norm(vel[:3])
        return self._object_on_ground() and speed < threshold

    def _simulate_until_settled(self, max_time: float = 5.0,
                                 record: bool = True) -> np.ndarray | None:
        """Run simulation until object settles or max_time exceeded.

        Returns trajectory array (T, 6) if record=True, else None.
        """
        steps_per_frame = max(1, int(1.0 / (TRAJECTORY_HZ * self.timestep)))
        max_steps = int(max_time / self.timestep)
        trajectory = [] if record else None

        for i in range(max_steps):
            self._step()
            if record and i % steps_per_frame == 0:
                trajectory.append(self._get_object_state())
            if self._object_settled():
                break

        if record:
            return np.array(trajectory) if trajectory else np.zeros((0, 6))
        return None

    def _simulate_until_ground(self, max_time: float = 5.0,
                                record: bool = True,
                                extra_time: float = 0.0) -> np.ndarray | None:
        """Run simulation until object hits ground (+ optional extra time).

        Returns trajectory array (T, 6) if record=True, else None.
        """
        steps_per_frame = max(1, int(1.0 / (TRAJECTORY_HZ * self.timestep)))
        max_steps = int(max_time / self.timestep)
        extra_steps = int(extra_time / self.timestep)
        trajectory = [] if record else None
        hit_ground = False
        steps_after_ground = 0

        for i in range(max_steps):
            self._step()
            if record and i % steps_per_frame == 0:
                trajectory.append(self._get_object_state())

            if not hit_ground and self._object_on_ground():
                hit_ground = True

            if hit_ground:
                steps_after_ground += 1
                if steps_after_ground >= extra_steps:
                    break

        if record:
            return np.array(trajectory) if trajectory else np.zeros((0, 6))
        return None

    def _simulate_for(self, duration: float, record: bool = True) -> np.ndarray | None:
        """Run simulation for a fixed duration.

        Returns trajectory array (T, 6) if record=True, else None.
        """
        steps_per_frame = max(1, int(1.0 / (TRAJECTORY_HZ * self.timestep)))
        n_steps = int(duration / self.timestep)
        trajectory = [] if record else None

        for i in range(n_steps):
            self._step()
            if record and i % steps_per_frame == 0:
                trajectory.append(self._get_object_state())

        if record:
            return np.array(trajectory) if trajectory else np.zeros((0, 6))
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_probe(self, probe_type: str) -> ProbeResult:
        """Execute a probe action and return structured observations.

        Dispatches to the appropriate probe controller.
        """
        from tossing.probes import get_probe
        controller = get_probe(probe_type)
        result = controller.execute(self)
        # Reset object back to gripper for next action
        self._soft_reset()
        return result

    def throw(self, params: ThrowParams) -> ThrowResult:
        """Execute a throw and evaluate success.

        Args:
            params: ThrowParams with theta (degrees), v (m/s), dt (s offset)
        """
        theta_rad = np.radians(params.theta)
        vx = params.v * np.cos(theta_rad)
        vz = params.v * np.sin(theta_rad)

        # Release with computed velocity
        self._release(linear_vel=np.array([vx, 0.0, vz]))

        # Simulate until settled
        trajectory = self._simulate_until_settled(max_time=5.0, record=True)

        # Evaluate: check if object is inside the basket
        final_pos = self._get_object_pos()
        basket_pos = self.data.xpos[self.basket_body_id].copy()

        # Distance in xz plane from basket center
        dx = final_pos[0] - basket_pos[0]
        dz = final_pos[2] - (basket_pos[2] + BASKET_HEIGHT)  # top of basket
        dist_horizontal = abs(dx)

        # Success if object is inside basket horizontally and vertically
        # Object x within basket radius, and z is between basket floor and top
        in_x = dist_horizontal < BASKET_RADIUS
        in_z = (basket_pos[2] <= final_pos[2] <= basket_pos[2] + 2 * BASKET_HEIGHT)
        success = in_x and in_z

        distance_to_center = np.sqrt(dx ** 2 + (final_pos[2] - basket_pos[2] - BASKET_HEIGHT) ** 2)

        flight_time = 0.0
        if trajectory is not None and len(trajectory) > 0:
            flight_time = len(trajectory) / TRAJECTORY_HZ

        result = ThrowResult(
            success=success,
            distance_to_basket=distance_to_center,
            trajectory=trajectory,
            flight_time=flight_time,
        )

        self._soft_reset()
        return result

    def render(self, width: int = 640, height: int = 480) -> dict[str, np.ndarray]:
        """Render side and top-down views.

        Returns dict with 'side' and 'top' keys, each an (H, W, 3) uint8 array.
        """
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height, width)

        # Side view camera: looking from -y axis at the xz plane
        side_cam = mujoco.MjvCamera()
        side_cam.lookat[:] = [self.basket_distance / 2, 0, LAUNCHER_HEIGHT / 2]
        side_cam.distance = max(4.0, self.basket_distance + 1.0)
        side_cam.azimuth = 90  # look from -y toward +y
        side_cam.elevation = -15

        self._renderer.update_scene(self.data, camera=side_cam)
        side_img = self._renderer.render().copy()

        # Top-down camera
        top_cam = mujoco.MjvCamera()
        top_cam.lookat[:] = [self.basket_distance / 2, 0, 0]
        top_cam.distance = max(4.0, self.basket_distance + 1.0)
        top_cam.azimuth = 90
        top_cam.elevation = -90  # straight down

        self._renderer.update_scene(self.data, camera=top_cam)
        top_img = self._renderer.render().copy()

        return {"side": side_img, "top": top_img}

    def render_pair_array(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Side+top stacked as a single uint8 (H, 2W, 3) array."""
        views = self.render(width, height)
        return np.concatenate([views["side"], views["top"]], axis=1)

    def render_pair(self, width: int = 640, height: int = 480) -> Image.Image:
        """Render side-by-side image for VLM input."""
        return Image.fromarray(self.render_pair_array(width, height))

    # ------------------------------------------------------------------
    # Video recording
    # ------------------------------------------------------------------

    def start_recording(self, fps: int = 30):
        """Begin capturing side+top frames inside _step().

        Frames are sampled at ~fps by decimating against the sim timestep.
        Any previously buffered frames are discarded.
        """
        self._record_frames = []
        self._record_step_counter = 0
        self._record_steps_per_capture = max(1, int(round(1.0 / (fps * self.timestep))))
        self._recording = True

    def stop_recording(self) -> list[np.ndarray]:
        """Stop capturing and return the buffered frames; clears the buffer."""
        self._recording = False
        frames = self._record_frames
        self._record_frames = []
        self._record_step_counter = 0
        return frames

    def save_recording(self, path, fps: int = 30) -> int:
        """Stop recording and write buffered frames to an MP4. Returns frame count.

        No-op if no frames were captured.
        """
        import imageio.v2 as imageio
        from pathlib import Path

        frames = self.stop_recording()
        if not frames:
            return 0
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(path), frames, fps=fps, codec="libx264",
                         macro_block_size=1)
        return len(frames)

    @property
    def object_pos(self) -> np.ndarray:
        return self._get_object_pos()

    @property
    def sim_time(self) -> float:
        return self.data.time
