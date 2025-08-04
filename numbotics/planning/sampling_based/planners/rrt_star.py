import numpy as np

from numbotics.planning.sampling_based import (
    SamplingPlannerBase,
    PlannerParams,
    StateSpace,
    Connector,
    PlanningGraph,
)



class RRTStar(SamplingPlannerBase):

    def __init__(
        self, space: StateSpace, connector: Connector, params: PlannerParams
    ):
        super().__init__(space, connector, params, graph=PlanningGraph(space.dimension, directed=True))


    def connection_radius(self):
        dim = float(self._space.dimension)
        k = float(self._graph._G.number_of_nodes())
        return self._params.rewire_factor * (np.log(dim) / dim) ** (1 / k)
    

    def plan(self):
        if self._start is None:
            raise ValueError("Must set start state before planning")
        if len(self._goals) == 0:
            raise ValueError("Must set goal states before planning")

        for i in range(self._params.max_iters):

            rand_state = self.sample_state()
            nearest = self._graph.nearest(rand_state)

            new_state = self._connector.steer(nearest.state, rand_state, distance_func=self._space.distance)
            if new_state is None:
                continue
            new_node = self._graph.add_vertex(new_state, np.inf)

            radius = self.connection_radius()
            neighbors = self._graph.k_nearest(
                new_node.state, radius=radius, k=self._params.k_nearest
            )

            best_parent = nearest
            best_cost = nearest.cost + self._space.distance(nearest.state, new_node.state)

            edges = {}
            for neighbor in neighbors:
                connect_edge = self._connector.connect(neighbor.state, new_node.state, distance_func=self._space.distance)
                if connect_edge is not None:
                    edges[neighbor.id] = connect_edge
                    cost = neighbor.cost + self._space.distance(neighbor.state, new_node.state)
                    if cost < best_cost:
                        best_parent = neighbor
                        best_cost = cost

            for goal in self._graph._goals:
                if self._space.distance(new_node.state, goal.state) < self._params.goal_tolerance:
                    self._graph.remove_node(new_node)
                    self._graph.add_edge(
                        best_parent,
                        goal,
                        weight=self._space.distance(new_node.state, goal.state),
                        update_cost=True,
                    )
                    break
            else:
                self._graph.add_edge(
                    best_parent,
                    new_node,
                    weight=self._space.distance(best_parent.state, new_node.state),
                    update_cost=True,
                )

                for neighbor in neighbors:
                    connect_edge = edges.get(neighbor.id, None)
                    if connect_edge is not None:
                        self._graph.rewire(
                            new_node,
                            neighbor,
                            self._space.distance(new_node.state, neighbor.state),
                        )
