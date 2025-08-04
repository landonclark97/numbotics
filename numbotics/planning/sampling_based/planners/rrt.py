import numpy as np

from numbotics.planning.sampling_based import (
    SamplingPlannerBase,
    PlannerParams,
    StateSpace,
    Connector,
    PlanningGraph,
)



class RRT(SamplingPlannerBase):

    def __init__(self, space: StateSpace, connector: Connector, params: PlannerParams):
        super().__init__(space, connector, params, graph=PlanningGraph(space.dimension, directed=True))


    def plan(self):
        if self._start is None:
            raise ValueError("Must set start state before planning")
        if len(self._goals) == 0:
            raise ValueError("Must set goal states before planning")

        for _ in range(self._params.max_iters):
            rand_state = self.sample_state()
            nearest = self._graph.nearest(rand_state)

            new_state = self._connector.steer(nearest.state, rand_state, distance_func=self._space.distance)
            if new_state is None:
                continue
            new_node = self._graph.add_vertex(new_state, np.inf)

            for goal in self._graph._goals:
                if self._space.distance(new_node.state, goal.state) < self._params.goal_tolerance:
                    self._graph.remove_node(new_node)
                    self._graph.add_edge(
                        nearest,
                        goal,
                        weight=self._space.distance(new_node.state, goal.state),
                        update_cost=False,
                    )
                    break
            else:
                self._graph.add_edge(
                    nearest,
                    new_node,
                    weight=self._space.distance(nearest.state, new_state),
                    update_cost=False,
                )
                continue
            break
