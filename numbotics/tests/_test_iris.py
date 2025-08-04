import numbotics.physics as phys
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import networkx as nx
from numbotics.math.spatial import trans_mat
from numbotics.planning import IrisSolver, IrisParams
from numbotics.robots import Arm
from numbotics.math.geometry import Polytope
from numbotics.physics import World, GraphChain, Cube


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

del cube2

cube.position = np.array([1.0, 1.0, 0.25])
cube3.position = np.array([-1.0, 1.0, 0.25])

print(cube.name)
print(cube3.name)


arm.remove_collision_pair(
    'base_link', 'half_arm_1_link'
)
arm.remove_collision_pair(
    'half_arm_2_link', 'spherical_wrist_1_link'
)
arm.remove_collision_pair(
    'base_link',
    cube3.name,
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


def min_coll(q: np.ndarray):
    return min([(p.distance, p.subject, p.target) for p in arm.collisions(q)], key=lambda x: x[0])



# for _ in range(1000000):
#     q = np.random.uniform(chain.joint_limits[:, 0], chain.joint_limits[:, 1])
#     dist = min_coll(q)
#     print(q, dist)
#     if dist < 0.0:
#         arm._chain.configuration = q
#         world.step()
#         time.sleep(5.0)

# quit()
P = Polytope.from_aabb(chain.joint_limits)
print(P.aabb())
print(P.volume())
print(P.largest_inscribed_ellipse().aabb())
print(P.largest_inscribed_ellipse().volume())

# print(P.contains(np.zeros((chain.dof,))))
# print(P.contains(P.sample(samples=100)))

iris = IrisSolver(
    subject=arm,
    params=IrisParams(
        tau=0.5,
        hyperplane_method='zoh',
    ),
)

P = iris.solve(seed=np.zeros((chain.dof,)), P_base=Polytope.from_aabb(chain.joint_limits))

iris = IrisSolver(
    subject=arm,
    params=IrisParams(
        tau=0.5,
        admissible_collisions=1e-3,
        hyperplane_method='np2',
    ),
)

P = iris.solve(seed=P.cheby_center(), P_base=P)

for q in P.sample(samples=10000):
    dist = min_coll(q)
    if dist[0] < 0.0:
        arm._chain.configuration = q
        world.step()
        time.sleep(5.0)
