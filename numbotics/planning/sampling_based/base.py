from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import choice

import numpy as np

from .connectors import Connector
from .space import StateSpace
from .graph import PlanningGraph
from numbotics.utils import logger as nlog



@dataclass(frozen=True)
class PlannerParams:
    max_iters: int
    goal_bias: float = 0.1
    rewire_factor: float = 1.1
    k_nearest: int = 50
    goal_tolerance: float = 1e-6



class SamplingPlannerBase(ABC):

    def __init__(self, space: StateSpace, connector: Connector, params: PlannerParams, graph: PlanningGraph | None = None):
        self._space = space
        self._connector = connector
        self._params = params
        self._graph = graph if graph is not None else PlanningGraph(space.dimension)
        self._start = None
        self._goals = []


    @abstractmethod
    def plan(self):
        raise NotImplementedError
    

    def solution(self):
        try:
            return self._graph.shortest_path_to_goal()
        except:
            nlog.warning("No solution found")
            return None


    def add_start(self, start: np.ndarray):
        if not self._connector.is_valid(start):
            raise ValueError("Start state is invalid")
        if not np.all(start >= self._space.lower_bounds) or not np.all(start <= self._space.upper_bounds):
            raise ValueError("Start state is out of bounds")
        self._start = start
        self._graph.add_start(start)


    def add_goal(self, goal: np.ndarray):
        if not self._connector.is_valid(goal):
            raise ValueError("Goal state is invalid")
        if not np.all(goal >= self._space.lower_bounds) or not np.all(goal <= self._space.upper_bounds):
            raise ValueError("Goal state is out of bounds")
        self._goals.append(goal)
        self._graph.add_goal(goal)
        

    def sample_state(self):
        if self._start is None:
            raise ValueError("Start state not set")
        if len(self._goals) == 0:
            raise ValueError("Goal states not set")
        return choice(self._goals) if np.random.rand() < self._params.goal_bias else self._space.sample()

