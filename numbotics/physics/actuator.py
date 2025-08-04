from typing import Callable
from weakref import proxy

import numpy as np

import numbotics.physics as physics
from numbotics.math import adjoint



class Actuator:

    def __init__(self, body: 'physics.PhysicsObject | physics.Link', control_law: Callable, local_offset: np.ndarray = np.eye(4)):
        self._body = proxy(body)
        self._control_law = control_law
        self._local_offset = local_offset

        physics.World().add_callback(self)


    def __call__(self):
        u = self._control_law()
        if u.shape[0] != 6:
            raise ValueError(f"Control law must return a 6D wrench vector, got {u.shape[0]}D.")
        
        T = self._body.pose @ self._local_offset
        u = adjoint(T).T @ u

        self._body.apply_wrench(u, local=True)
