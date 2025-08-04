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



class Ellipse:

    def __init__(self, C: np.ndarray, d: np.ndarray):
        if C.ndim != 2:
            raise ValueError("C must be a 2D array")
        if d.ndim != 1:
            raise ValueError("d must be a 1D array")
        if C.shape[1] != d.shape[0]:
            raise ValueError("C must have the same number of columns as d")
        if not is_PSD(C.T @ C):
            raise ValueError("C must be positive semidefinite")

        self._C = C
        self._d = d


    def __call__(self, x: np.ndarray | cp.Variable):
        if not (isinstance(x, cp.Variable) or isinstance(x, np.ndarray)):
            raise ValueError("x must be a numpy array or a cvxpy variable")
        if isinstance(x, cp.Variable):
            return cp.quad_form(x - self.d, self.C.T @ self.C) <= 1.0
        return (x - self.d).T @ self.C.T @ self.C @ (x - self.d) <= 1.0
    

    @property
    def C(self):
        return self._C


    @property
    def d(self):
        return self._d
    

    @property
    def m(self):
        return self._C.shape[0]


    @property
    def n(self):
        return self._C.shape[1]
   

    def scale(self, scale: float):
        if scale <= 0:
            raise ValueError("scale must be positive")
        return Ellipse(self.C / np.power(scale, 1.0 / self.n),
                       self.d / scale)
    

    def contains(self, x: np.ndarray):
        initial_shape = x.shape
        if x.shape[-1] != self.n:
            raise ValueError("x must have the same dimension as the polytope")
        contains = ((x.reshape(-1, self.n, 1) - self.d[None,...,None]).T @ self.C.T @ self.C @ (x.reshape(-1, self.n, 1) - self.d[None,...,None])).squeeze(-1) <= 1.0
        if len(initial_shape) == 1:
            return contains[0]
        return contains.reshape(initial_shape[:-1])
    

    def volume(self):
        if self.n == 0:
            return 0.0
        if self.m < self.n:
            return np.inf
        return (np.pi ** (float(self.n) / 2.0) / gamma((float(self.n) / 2.0) + 1.0)) / np.abs(np.linalg.det(self.C))
    

    def aabb(self):
        C_sqrt = np.sqrt(np.diag(np.linalg.inv(self.C.T @ self.C)))
        return np.vstack((self.d - C_sqrt, self.d + C_sqrt)).T