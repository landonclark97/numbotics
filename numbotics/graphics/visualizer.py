from typing import Union

import numpy as np
import meshcat
import meshcat.geometry as g

from numbotics.utils import pipes


VisualShape = Union[g.Box, g.Mesh, g.Sphere, g.Cylinder, g.Plane]



class Visualizer:

    def __init__(self):
        with pipes():
            self.vis = meshcat.Visualizer()
            self.vis.open()
        self.vis["/Background"].set_property("top_color", [0.35, 0.55, 0.75])
        self.vis["/Background"].set_property("bottom_color", [0.55, 0.55, 0.55])
        self.vis["/Cameras/default/rotated/<object>"].set_property("far", 10000)


    def __del__(self):
        self.close()


    def close(self):
        self.vis.delete()
        self.vis = None


    def url(self):
        return self.vis.url()


    def register(self, name: str, visual_shape: VisualShape):
        self.vis[name].set_object(visual_shape)


    def set_transform(self, name: str, transform: np.ndarray):
        self.vis[name].set_transform(transform)


    def set_color(self, name: str, color: np.ndarray):
        self.vis[name].set_property("color", color.tolist())


    def set_alpha(self, name: str, alpha: float):
        self.vis[name].set_property("opacity", alpha)


    def set_visible(self, name: str, visible: bool):
        self.vis[name].set_property("visible", visible)


    def unregister(self, name: str):
        self.vis[name].delete()

        
