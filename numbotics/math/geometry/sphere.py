from typing import Literal

import numpy as np
import cvxpy as cp
from scipy.special import gamma

from numbotics.math import is_PSD


SCALE_MODES = Literal["best", "fast"]

if cp.MOSEK in cp.installed_solvers():
    SDP_SOLVER = cp.MOSEK
else:
    SDP_SOLVER = cp.CLARABEL



class Sphere:

    def __init__(self, d: np.ndarray, r: float):
        if d.ndim != 1:
            raise ValueError("d must be a 1D array")
        if r <= 0:
            raise ValueError("r must be positive")

        self._d = d
        self._r = r


    def __call__(self, x: np.ndarray | cp.Variable):
        if not (isinstance(x, cp.Variable) or isinstance(x, np.ndarray)):
            raise ValueError("x must be a numpy array or a cvxpy variable")
        if isinstance(x, cp.Variable):
            return cp.norm(x - self.d, axis=-1) <= self.r
        return np.linalg.norm(x - self.d) <= self.r
    

    @property
    def d(self):
        return self._d
    

    @property
    def r(self):
        return self._r
    

    @property
    def n(self):
        return self._d.shape[0]
   

    def scale(self, scale: float):
        if scale <= 0:
            raise ValueError("scale must be positive")
        return Sphere(self.d, self.r / np.power(scale, 1.0 / self.n))
    

    def contains(self, x: np.ndarray):
        initial_shape = x.shape
        if x.shape[-1] != self.n:
            raise ValueError("x must have the same dimension as the sphere")
        contains = np.linalg.norm(x.reshape(-1, self.n, 1) - self.d[None,...,None], axis=-1) <= self.r
        if len(initial_shape) == 1:
            return contains[0]
        return contains.reshape(initial_shape[:-1])
    

    def volume(self):
        if self.n == 0:
            return 0.0
        return (np.pi ** (float(self.n) / 2.0) / gamma((float(self.n) / 2.0) + 1.0)) * np.power(self.r, self.n)
    

    def aabb(self):
        return np.vstack((self.d - self.r, self.d + self.r))