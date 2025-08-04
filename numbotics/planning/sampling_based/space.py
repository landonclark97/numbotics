from typing import Callable
from abc import ABC, abstractmethod

import numpy as np



class StateSpace(ABC):
    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray, sampler: Callable[[], np.ndarray] | None = None):
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds
        if self._lower_bounds.shape != self._upper_bounds.shape:
            raise ValueError("Lower and upper bounds must have the same shape")
        if self._lower_bounds.ndim != 1:
            raise ValueError("Lower and upper bounds must be 1D arrays")
        self._sampler = sampler


    def sample(self):
        if self._sampler is None:
            return np.random.uniform(self.lower_bounds, self.upper_bounds)
        return self._sampler()
    

    @abstractmethod
    def distance(self, state1: np.ndarray, state2: np.ndarray):
        raise NotImplementedError
    

    @property
    def lower_bounds(self):
        return self._lower_bounds


    @property
    def upper_bounds(self):
        return self._upper_bounds


    @property
    def dimension(self):
        return self._lower_bounds.shape[0]


    @property
    def volume(self):
        return np.prod(self._upper_bounds - self._lower_bounds)