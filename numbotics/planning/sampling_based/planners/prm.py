import numpy as np

from numbotics.planning.sampling_based import (
    SamplingPlannerBase,
    PlannerParams,
    StateSpace,
    Connector,
    PlanningGraph,
)



class PRM(SamplingPlannerBase):

    def __init__(self, space: StateSpace, connector: Connector, params: PlannerParams):
        super().__init__(space, connector, params, graph=PlanningGraph(space.dimension, directed=False))
    

    def plan(self):
        if self._start is None:
            raise ValueError("Must set start state before planning")
        if len(self._goals) == 0:
            raise ValueError("Must set goal states before planning")

        for _ in range(self._params.max_iters):
            rand_state = self.sample_state()

            for goal in self._graph._goals:
                if self._space.distance(rand_state, goal.state) < self._params.goal_tolerance:
                    new_node = goal
                    break
            else:
                new_node = self._graph.add_vertex(rand_state, np.inf)

            neighbors = self._graph.k_nearest(
                new_node.state, radius=np.inf, k=self._params.k_nearest
            )

            for neighbor in neighbors:
                connect_edge = self._connector.connect(neighbor.state, new_node.state, distance_func=self._space.distance)
                if connect_edge is not None:
                    self._graph.add_edge(
                        neighbor,
                        new_node,
                        weight=self._space.distance(neighbor.state, new_node.state),
                        update_cost=False,
                    )
