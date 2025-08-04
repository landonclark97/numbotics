import numbotics.physics as phys
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import networkx as nx
from typing import Callable
from numbotics.math.spatial import trans_mat
from numbotics.planning import IrisSolver, IrisParams
from numbotics.robots import Arm
from numbotics.math.geometry import Polytope
from numbotics.physics import World, GraphChain, Cube
from numbotics.planning.sampling_based import StateSpace, DiscreteConnector, ConnectorParams, PlannerParams, ContinuousConnector
from numbotics.planning.sampling_based.planners import RRT, RRTStar, PRM, PRMStar


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()


world = phys.World(visualize=args.vis)
world.gravity = np.array([0, 0, 0])

# chain = GraphChain.from_urdf('./numbotics/tests/models/kortex/kinova_no_obs.urdf')
chain = GraphChain.from_urdf('./numbotics/tests/models/kortex/kinova_cyl.urdf')
arm = Arm(chain)

cube = Cube(half_extent=0.5, mass=1.0)
cube2 = Cube(half_extent=0.5, mass=1.0)
cube3 = Cube(half_extent=0.5, mass=1.0)

cube.position = np.array([1.35, -1.5, 0.25])
cube2.position = np.array([1.35, 1.5, 0.25])
cube3.position = np.array([1.35, 0.0, 1.1])


arm.remove_collision_pair(
    'base_link', 'half_arm_1_link'
)
arm.remove_collision_pair(
    'half_arm_2_link', 'spherical_wrist_1_link'
)

from itertools import combinations
links = [
    'spherical_wrist_1_link',
    'spherical_wrist_2_link',
    'bracelet_link',
    'end_effector_link',
    'camera_link',
    'camera_depth_frame',
    'camera_color_frame',
    'tool_frame',
    'robotiq_arg2f_base_link',
    'gripper',
    'camera',
]

for link_a, link_b in combinations(links, 2):
    arm.remove_collision_pair(link_a, link_b)


world.step()

# for _ in range(10000000):
#     q = np.random.uniform(arm.joint_limits[:, 0], arm.joint_limits[:, 1])
#     arm._chain.configuration = q
#     world.update_visualizer()
#     if arm.in_collision(q):
#         time.sleep(2.0)

def min_coll(q: np.ndarray):
    return min([(p.distance, p.subject, p.target) for p in arm.collisions(q)], key=lambda x: x[0])

class LinearInterpolator:
    def __init__(self, start: np.ndarray, goal: np.ndarray):
        self.start = start
        self.goal = goal

    def __call__(self, t: float) -> np.ndarray:
        return (1.0 - t) * self.start + t * self.goal


def to_trajectory(start: np.ndarray, goal: np.ndarray) -> Callable[[float], np.ndarray]:
    return LinearInterpolator(start, goal)

from numbotics.planning import unit_bspline


def validity_checker(q: np.ndarray):
    if arm.in_collision(q):
        return False
    return True

c_params = ConnectorParams(
    resolution=1e-1,
    max_distance=np.pi,
    # trajectory_func=unit_bspline,
    validity_checker=validity_checker,
)

connector = DiscreteConnector(c_params)

cont_params = ConnectorParams(
    resolution=5e-2,
    max_distance=np.pi / 2.0,
    # trajectory_func=unit_bspline,
    validity_checker=lambda q: arm.closest_to(q).distance,
)
cont_connector = ContinuousConnector(cont_params)


p_params = PlannerParams(
    max_iters=100,
    goal_bias=0.1,
    k_nearest=10,
)


class Rn(StateSpace):
    def distance(self, state1: np.ndarray, state2: np.ndarray):
        return np.linalg.norm(state1 - state2)

space = Rn(
    lower_bounds=arm.joint_limits[:, 0],
    upper_bounds=arm.joint_limits[:, 1],
)

planner = PRM(
    space=space,
    connector=connector,
    params=p_params,
)


goal = np.array([0.0, np.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
start = np.zeros((arm.dof,)) # np.array([0.0, np.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

planner.add_start(start)
planner.add_goal(goal)

import time
s_time = time.time()
planner.plan()
e_time = time.time()
print(f"Time taken: {e_time - s_time} seconds")

path = planner.solution()
if path is None:
    print("No path found")
    exit()

states = np.array([node.state for node in path])
print(f'path length: {np.linalg.norm(np.diff(states, axis=0), axis=1).sum()}')

def linear_interp(states: np.ndarray, samples: int):
    lengths = np.hstack((0.0, np.linalg.norm(np.diff(states, axis=0), axis=1)))
    lengths = np.cumsum(lengths) 
    lengths = lengths / np.amax(lengths)
    ts = np.linspace(0.0, 1.0, samples)
    return np.array([np.interp(ts, lengths, states[:, i]) for i in range(states.shape[1])]).T


interp_states = linear_interp(states, 250)

if args.vis:
    while True:
        for state in interp_states:
            arm._chain.configuration = state
            world.update_visualizer()
            time.sleep(1.0 / 60.0)







