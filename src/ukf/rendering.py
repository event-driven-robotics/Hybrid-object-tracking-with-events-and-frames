"""Optional PyRender-based comparison renderer.

This module is intentionally not imported by the filtering pipeline.
"""

from __future__ import annotations

import numpy as np


class PoseComparisonRenderer:
    """Render pose-only and pose-plus-velocity estimates next to each other."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int, mesh_path: str):
        try:
            import pyrender
            import trimesh
        except ImportError as error:
            raise ImportError("Rendering requires pyrender and trimesh.") from error
        self._pyrender = pyrender
        self.width, self.height = width, height
        self.scene = pyrender.Scene(bg_color=(0.0, 0.0, 0.0))
        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
        self.scene.add(camera, pose=np.diag((1.0, -1.0, -1.0, 1.0)))
        self.scene.add(pyrender.PointLight(intensity=50.0))
        mesh = pyrender.Mesh.from_trimesh(trimesh.load(mesh_path))
        self.mesh_node = pyrender.Node(mesh=mesh, matrix=np.eye(4))
        self.scene.add_node(self.mesh_node)
        self.renderer = pyrender.OffscreenRenderer(width, height)

    def render(self, velocity_pose: np.ndarray, pose_only: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Return a side-by-side BGR comparison image; does not open a window."""
        import cv2

        left = self._render_on_background(pose_only, background, "pose only", (20, 160, 200))
        right = self._render_on_background(velocity_pose, background, "pose + velocity", (0, 0, 255))
        return np.hstack((left, right))

    def close(self) -> None:
        self.renderer.delete()

    def _render_on_background(self, pose: np.ndarray, background: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
        import cv2

        self.scene.set_pose(self.mesh_node, pose)
        rendered, _ = self.renderer.render(self.scene)
        rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        canvas[rendered > 0] = rendered[rendered > 0]
        cv2.putText(canvas, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return canvas
