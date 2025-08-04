from dataclasses import dataclass

import numpy as np
import numbotics.physics as physics
from numbotics.utils import parse_shape_kwargs, Shape



@dataclass(frozen=True)
class Contact:
    subject: 'physics.PhysicsObject | physics.Link'
    target: 'physics.PhysicsObject | physics.Link'
    position_on_subject: np.ndarray
    position_on_target: np.ndarray
    contact_normal_target_to_subject: np.ndarray
    contact_distance: float
    normal_force: float
    lateral_friction_a: float
    lateral_friction_a_dir: np.ndarray
    lateral_friction_b: float
    lateral_friction_b_dir: np.ndarray



@dataclass(frozen=True)
class Proximity:
    subject: 'physics.PhysicsObject | physics.Link'
    target: 'physics.PhysicsObject | physics.Link'
    position_on_subject: np.ndarray
    position_on_target: np.ndarray
    normal_target_to_subject: np.ndarray
    distance: float



class CollisionShape:

    def __init__(self, shape: Shape, **kwargs):
        if not isinstance(shape, Shape):
            raise ValueError(f"Invalid shape type: {shape}")
        self.shape = shape
        self._shape_info = parse_shape_kwargs(kwargs)[1]

    
    def register(self, **kwargs):
        self.col_id = self.shape.register_collision_shape(**self._shape_info, **kwargs)
