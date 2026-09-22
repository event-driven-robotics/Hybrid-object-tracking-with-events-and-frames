"""
The file is for rendering object with poses and the object mesh
"""

import cv2
import numpy as np
import os
import sys
import trimesh
import math
import json
import pyquaternion as Quaternion
from tqdm import tqdm

# Force rendering with EGL
os.environ["PYOPENGL_PLATFORM"] = "egl"
import pyrender


class rendering_object():
    def __init__(self, camera_path, mesh_ply_path):
        # Create the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])

        # Load camera parameters, create and insert the camera
        with open(camera_path, 'r', encoding='utf-8') as f:
            # print(info_list[i])
            info_json = json.load(f)
        self.fx = info_json['focalX']
        self.fy = info_json['focalY']
        self.cx = info_json['centerX']
        self.cy = info_json['centerY']
        self.camera_width = info_json['width']
        self.camera_height = info_json['height']
        self.camera = pyrender.IntrinsicsCamera(self.fx, self.fy, self.cx, self.cy)
        self.scene.add(self.camera)

        # Create and insert the light
        self.light = pyrender.PointLight(intensity=50.0)
        self.scene.add(self.light)

        # Load the mesh
        self.mesh = pyrender.Mesh.from_trimesh(
            trimesh.load(mesh_ply_path))

        # Create and insert a node for the object
        self.mesh_node = pyrender.Node(mesh=self.mesh, matrix=np.eye(4))
        self.scene.add_node(self.mesh_node)

        # Create a renderer
        self.renderer = pyrender.OffscreenRenderer(self.camera_width, self.camera_height)

    def rendering(self, object_pose, background):
        # Render the scene
        self.scene.set_pose(self.mesh_node, object_pose)
        render_rgb, render_depth = self.renderer.render(self.scene)
        render_rgb = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)

        # Showing with a background
        background = background.astype(np.uint8)
        non_zero_render = render_rgb > 0
        background[non_zero_render] = render_rgb[non_zero_render]
        cv2.imshow('display', background)
        cv2.waitKey(1)
        # cv2.imwrite("./test-renderer_output.png", render_rgb)

