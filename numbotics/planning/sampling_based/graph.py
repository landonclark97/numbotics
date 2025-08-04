from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import networkx as nx

from numbotics.math.geometry import ApproximateNearestNeighborsIndex



@dataclass(frozen=True)
class Node:
    id: str
    state: np.ndarray
    cost: float = np.inf

    def __post_init__(self):
        if not self.id.startswith("v_") and not self.id.startswith("g_"):
            raise ValueError(f"Invalid node ID: {self.id}")



@dataclass(frozen=True)
class Edge:
    u: Node
    v: Node
    weight: float
    params: dict



class PlanningGraph:

    def __init__(self, dimension: int, directed: bool = False):
        self._knn = ApproximateNearestNeighborsIndex(dimension)
        self._G = nx.DiGraph() if directed else nx.Graph()
        self._goals = []


    def __len__(self):
        return len(self._knn)


    def add_start(self, state: np.ndarray) -> Node:
        if self._G.has_node("v_0"):
            raise ValueError("Start node already exists")
        id = self._knn.add_point(state)
        assert id == 0, "Start node must be the first node"
        self._G.add_node("v_0", state=state, cost=0)
        return Node(id="v_0", state=state, cost=0)


    def add_vertex(self, state: np.ndarray, cost: float = np.inf) -> Node:
        id = self._knn.add_point(state)
        node_id = f"v_{id}"
        self._G.add_node(node_id, state=state, cost=cost)
        return Node(id=node_id, state=state, cost=cost)


    def add_goal(self, state: np.ndarray, cost: float = np.inf) -> Node:
        node_id = f"g_{len(self._goals)}"
        self._G.add_node(node_id, state=state, cost=cost)
        self._goals.append(Node(id=node_id, state=state, cost=cost))
        return self._goals[-1]


    def add_edge(self, u: Node, v: Node, weight: float, params: dict = {}, update_cost: bool = True):
        self._G.add_edge(u.id, v.id, weight=weight, params=params)
        if update_cost:
            self.update_costs_recursive(v)
        return Edge(u, v, weight, params)


    def get_node(self, node_id: str) -> Node:
        if self._G.has_node(node_id):
            return Node(
                id=node_id,
                state=self._G.nodes[node_id]["state"],
                cost=self._G.nodes[node_id]["cost"],
            )
        else:
            raise ValueError(f"Invalid node ID: {node_id}")
        

    def get_edge(self, u_id: str, v_id: str) -> Edge:
        if self._G.has_edge(u_id, v_id):
            return Edge(
                u=self.get_node(u_id),
                v=self.get_node(v_id),
                weight=self._G[u_id][v_id]["weight"],
                params=self._G[u_id][v_id].get("params", {}),
            )
        else:
            raise ValueError(f"Invalid edge: ({u_id} -> {v_id})")


    def children(self, node: Node) -> List[Node]:
        children = []
        for child in self._G.successors(node.id):
            children.append(self.get_node(child))
        return children
    

    def parents(self, node: Node) -> List[Node]:
        parents = []
        for parent in self._G.predecessors(node.id):
            parents.append(self.get_node(parent))
        return parents


    def descendants(self, node: Node) -> List[Node]:
        descs = []
        for desc in nx.descendants(self._G, node.id):
            descs.append(self.get_node(desc))
        return descs
    

    def ancestors(self, node: Node) -> List[Node]:
        ancestors = []
        for ancestor in nx.ancestors(self._G, node.id):
            ancestors.append(self.get_node(ancestor))
        return ancestors


    def node_cost(self, node: Node) -> float:
        return self._G.nodes[node.id]["cost"]


    def node_state(self, node: Node) -> np.ndarray:
        return self._G.nodes[node.id]["state"]


    def update_node_cost(self, node: Node, cost: float):
        self._G.nodes[node.id]["cost"] = cost


    def path_cost(self, path: List[Node]) -> float:
        return nx.path_weight(self._G, [node.id for node in path], weight="weight")


    def remove_node(self, node: Node):
        self._G.remove_node(node.id)
        self._knn.remove_points([int(node.id.split("_")[1])])


    def remove_edge(self, from_node: Node, to_node: Node):
        self._G.remove_edge(from_node.id, to_node.id)


    def detach_parent(self, node: Node):
        parents = self.parents(node)
        if len(parents) != 1:
            raise ValueError(f"Nodes should have one parent, found {len(parents)}")
        parent = parents[0]
        self.remove_edge(parent, node)


    def shortest_path(
        self, source: Node, target: Node
    ) -> List[Node]:
        path = nx.shortest_path(self._G, source.id, target.id, weight="weight")
        return [self.get_node(p) for p in path]


    def k_nearest(
        self, state: np.ndarray, k: int, radius: float = np.inf
    ) -> List[Node]:
        q_near_all, near_inds = self._knn.k_nearest(state, k=k, return_labels=True)
        mask = np.linalg.norm(q_near_all - state, axis=1) < radius
        nodes = []
        for idx in near_inds[mask]:
            nodes.append(self.get_node(f"v_{idx}"))
        return nodes


    def nearest(self, state: np.ndarray) -> Node:
        _, idx = self._knn.nearest(state, return_labels=True)
        return self.get_node(f"v_{idx}")
    

    def rewire(self, parent: Node, child: Node, weight: float):
        if child.id.startswith("g_"):
            return
        if self.node_cost(parent) + weight < self.node_cost(child):
            self.detach_parent(child)
            self.add_edge(parent, child, weight=weight)
            self.update_costs_recursive(child)


    def update_costs_recursive(self, node: Node, base_cost: float | None = None):
        if base_cost is None:
            base_cost = nx.shortest_path_length(self._G, "v_0", node.id, weight="weight")
        self.update_node_cost(node, base_cost)
        for child in self.children(node):
            edge_weight = self.get_edge(node, child).weight
            self.update_costs_recursive(child, base_cost + edge_weight)


    def shortest_path_to_goal(self) -> Optional[List[Node]]:

        if len(self._goals) == 0:
            return None

        min_cost = np.inf
        best_path = None

        for goal_node in self._goals:
            try:
                path_cost = nx.shortest_path_length(
                    self._G, source="v_0", target=goal_node.id, weight="weight"
                )
                if path_cost < min_cost:
                    min_cost = path_cost
                    best_path = nx.shortest_path(
                        self._G, source="v_0", target=goal_node.id, weight="weight"
                    )
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            return None

        nodes = []
        for node_id in best_path:
            nodes.append(
                Node(
                    id=node_id,
                    state=self._G.nodes[node_id]["state"],
                    cost=self._G.nodes[node_id]["cost"],
                )
            )

        return nodes
