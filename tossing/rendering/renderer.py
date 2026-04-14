"""Offscreen rendering for VLM input images.

Produces side-view and top-down views of the tossing scene using MuJoCo's
built-in renderer with EGL backend.
"""

from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image


class SceneRenderer:
    """Renders side and top-down views of a TossEnv scene."""

    def __init__(self, model: mujoco.MjModel, width: int = 640, height: int = 480):
        self.model = model
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height, width)

    def render_side_view(self, data: mujoco.MjData,
                         lookat: np.ndarray | None = None,
                         distance: float = 4.0) -> np.ndarray:
        """Render side view (from -y axis, looking at xz plane).

        Returns (H, W, 3) uint8 array.
        """
        cam = mujoco.MjvCamera()
        if lookat is not None:
            cam.lookat[:] = lookat
        else:
            cam.lookat[:] = [1.0, 0, 0.75]
        cam.distance = distance
        cam.azimuth = 90
        cam.elevation = -15

        self._renderer.update_scene(data, camera=cam)
        return self._renderer.render().copy()

    def render_top_view(self, data: mujoco.MjData,
                        lookat: np.ndarray | None = None,
                        distance: float = 4.0) -> np.ndarray:
        """Render top-down view.

        Returns (H, W, 3) uint8 array.
        """
        cam = mujoco.MjvCamera()
        if lookat is not None:
            cam.lookat[:] = lookat
        else:
            cam.lookat[:] = [1.0, 0, 0]
        cam.distance = distance
        cam.azimuth = 90
        cam.elevation = -90

        self._renderer.update_scene(data, camera=cam)
        return self._renderer.render().copy()

    def render_pair(self, data: mujoco.MjData, **kwargs) -> Image.Image:
        """Render side + top views concatenated horizontally.

        Returns a PIL Image suitable for VLM input.
        """
        side = self.render_side_view(data, **kwargs)
        top = self.render_top_view(data, **kwargs)
        combined = np.concatenate([side, top], axis=1)
        return Image.fromarray(combined)

    def render_closeup(self, data: mujoco.MjData,
                       body_pos: np.ndarray,
                       distance: float = 0.5) -> np.ndarray:
        """Render a close-up view of a specific body position.

        Useful for showing the object on the gripper to the VLM.
        Returns (H, W, 3) uint8 array.
        """
        cam = mujoco.MjvCamera()
        cam.lookat[:] = body_pos
        cam.distance = distance
        cam.azimuth = 90
        cam.elevation = -10

        self._renderer.update_scene(data, camera=cam)
        return self._renderer.render().copy()

    def close(self):
        """Clean up renderer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
