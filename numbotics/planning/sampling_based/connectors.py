from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

from numbotics.planning import unit_bspline



@dataclass(frozen=True)
class ConnectorParams:
    resolution: float = 5e-2
    max_distance: float = 1.0
    trajectory_func: Callable[[np.ndarray, np.ndarray], Callable[[float], np.ndarray]] = lambda x, y : unit_bspline(np.array([x, y]))
    validity_checker: Callable[[np.ndarray], bool] | None = None

    def __post_init__(self):
        if self.resolution <= 0:
            raise ValueError("Resolution must be positive")
        if self.resolution < 0.0 or self.resolution >= 1.0:
            raise ValueError("Resolution must be strictly between 0.0 and 1.0")
        if self.max_distance <= 0:
            raise ValueError("Max distance must be positive")
        if self.validity_checker is None:
            raise ValueError("Validity checker must be provided")
        if self.trajectory_func is None:
            raise ValueError("Trajectory conversion function must be provided")



class Connector(ABC):

    @abstractmethod
    def connect(self, start: np.ndarray, goal: np.ndarray):
        raise NotImplementedError


    @abstractmethod
    def steer(self, start: np.ndarray, goal: np.ndarray):
        raise NotImplementedError


    @abstractmethod
    def is_valid(self, state: np.ndarray):
        raise NotImplementedError



class DiscreteConnector(Connector):

    def __init__(self, params: ConnectorParams):
        self._params = params


    def connect(
            self,
            start: np.ndarray,
            goal: np.ndarray,
            distance_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y)
        ):
        distance = distance_func(start, goal)
        if distance <= np.finfo(np.float32).eps:
            return None
        
        trajectory = self._params.trajectory_func(start, goal)
        T = np.arange(0.0, 1.0, self._params.resolution / distance)
        T = np.append(T, 1.0)
        
        for t in T:
            if not self.is_valid(trajectory(t)):
                return None
        
        return np.copy(goal)


    def steer(
            self,
            start: np.ndarray,
            goal: np.ndarray,
            distance_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y)
        ):
        distance = distance_func(start, goal)
        if distance <= np.finfo(np.float32).eps:
            return None
        
        if distance <= self._params.max_distance:
            T_f = 1.0
        else:
            T_f = self._params.max_distance / distance
        trajectory = self._params.trajectory_func(start, goal)
        T = np.arange(0.0, T_f, self._params.resolution / distance)
        T = np.append(T, T_f)

        for t in T:
            if not self.is_valid(trajectory(t)):
                return None
        
        return np.copy(trajectory(T_f))
    

    def is_valid(self, state: np.ndarray):
        return self._params.validity_checker(state)    



class ContinuousConnector(Connector):

    def __init__(self, params: ConnectorParams):
        self._params = params


    def connect(
            self,
            start: np.ndarray,
            goal: np.ndarray,
            distance_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y)
        ):
        distance = distance_func(start, goal)
        if distance <= np.finfo(np.float32).eps:
            return None
        
        trajectory = self._params.trajectory_func(start, goal)
        T = np.arange(0.0, 1.0, self._params.resolution / distance)
        T = np.append(T, 1.0)

        for t0, t1 in zip(T[:-1], T[1:]):
            constraint = NonlinearConstraint(
                lambda t: self._params.validity_checker(trajectory(t[0])),
                lb=-np.inf,
                ub=0.0,
            )
            result = minimize(
                lambda t: t[0],
                x0=np.array([(t0 + t1) / 2.0]),
                bounds=[(t0, t1)],
                method='SLSQP',
                constraints=[constraint],
            )
            if result.success:
                return None
        
        return np.copy(goal)


    def steer(
            self,
            start: np.ndarray,
            goal: np.ndarray,
            distance_func: Callable[[np.ndarray, np.ndarray], float] = lambda x, y: np.linalg.norm(x - y)
        ):
        distance = distance_func(start, goal)
        if distance <= np.finfo(np.float32).eps:
            return None
        
        if distance <= self._params.max_distance:
            T_f = 1.0
        else:
            T_f = self._params.max_distance / distance
        trajectory = self._params.trajectory_func(start, goal)
        T = np.arange(0.0, T_f, self._params.resolution / distance)
        T = np.append(T, T_f)

        for t0, t1 in zip(T[:-1], T[1:]):
            constraint = NonlinearConstraint(
                lambda t: self._params.validity_checker(trajectory(t[0])),
                lb=-np.inf,
                ub=0.0,
            )
            result = minimize(
                lambda t: t[0],
                x0=np.array([(t0 + t1) / 2.0]),
                bounds=[(t0, t1)],
                method='SLSQP',
                constraints=[constraint],
            )
            if result.success:
                return None
        
        return np.copy(trajectory(T_f))
    

    def is_valid(self, state: np.ndarray):
        return self._params.validity_checker(state) > 0.0