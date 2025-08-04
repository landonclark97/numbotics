import numbotics.physics as phys

import numpy as np
from scipy.spatial.transform import Rotation as R
import networkx as nx

from numbotics.math.spatial import trans_mat

import time


world = phys.World(visualize=True)
world.gravity = np.array([0, 0, -9.81])


BT = np.eye(4)
BT[:3, 3] = np.array([0.0, 0.0, 0.0])
BT[:3, :3] = R.from_euler("y", np.pi / 2).as_matrix()
T = np.eye(4)
T[:3, 3] = np.array([1.0, 0.0, 0.0])
Tlink = np.eye(4)
Tlink[:3, 3] = np.array([0.5, 0.0, 0.0])
Tlink[:3, :3] = R.from_euler("y", -np.pi / 2.0).as_matrix()

link_list = [
    phys.BasicLink(np.eye(4), mass=5.0, geometry=phys.Geometry.CYLINDER, radius=0.1, height=1.0),
    phys.BasicLink(Tlink, mass=5.0, geometry=phys.Geometry.CYLINDER, radius=0.1, height=1.0),
    phys.BasicLink(Tlink, mass=5.0, geometry=phys.Geometry.CYLINDER, radius=0.1, height=1.0),
    phys.BasicLink(Tlink, mass=5.0, geometry=phys.Geometry.CYLINDER, radius=0.1, height=1.0),
]
joint_list = [
    phys.Joint(trans_mat(pos=np.array([0, 0, 0.5]), orn=R.from_euler("xy", [-np.pi / 2, -np.pi / 2]).as_matrix()), np.array([0, 0, 1]), phys.Constraint.REVOLUTE),
    phys.Joint(T, np.array([0, 0, 1]), phys.Constraint.REVOLUTE),
    phys.Joint(T, np.array([0, 0, 1]), phys.Constraint.REVOLUTE),
]

arm = phys.SerialChain(
    link_list,
    joint_list,
)
arm.base_position = np.array([5,5,5])

plane = phys.Plane(
    0, 
    np.array([0, 0, 1]), 
    static=True
)
plane.lateral_friction = 0.9
plane.rolling_friction = 1e-4
plane.spinning_friction = 1e-4

plane.position = np.array([0, 0, -1.25])

# Create a directed graph for our car structure
G = nx.DiGraph()

# Create transformation matrices for the car body and wheels
body_transform = np.eye(4)

# Create the car body (root node)
body = phys.BasicLink(body_transform, mass=1000.0, geometry=phys.Geometry.CUBOID, half_extents=np.array([1.0, 0.5, 0.2]), static=False)
G.add_node(0, link=body)

# Create wheels
wheel_radius = 0.2
wheel_width = 0.1
wheel_mass = 25.0

# Define wheel positions relative to the body
wheel_positions = [
    np.array([0.8, 0.6, -0.2]),  # Front right
    np.array([0.8, -0.6, -0.2]), # Front left
    np.array([-0.8, 0.6, -0.2]), # Rear right
    np.array([-0.8, -0.6, -0.2]) # Rear left
]

# Create wheels and add them to the graph
for i, pos in enumerate(wheel_positions, start=1):
    wheel_transform = trans_mat(orn=R.from_euler("x", np.pi / 2).as_matrix())
    wheel = phys.BasicLink(wheel_transform, mass=wheel_mass, geometry=phys.Geometry.CYLINDER, radius=wheel_radius, height=wheel_width, static=False)
    G.add_node(i, link=wheel)

    # Create revolute joints for the wheels
    joint = phys.Joint(trans_mat(pos=pos), np.array([0, 1, 0]), phys.Constraint.REVOLUTE)
    G.add_edge(0, i, joint=joint)

# Create the GraphChain
chain = phys.GraphChain(G, static_base=False)
for link in chain._links[1:]:
    link.rolling_friction = 0.01
    link.lateral_friction = 0.9
    link.spinning_friction = 0.5

world.add_constraint(
    chain._links[0], 
    arm._links[0], 
    phys.Joint(
        np.eye(4),
        np.array([0, 0, 0]), 
        phys.Constraint.FIXED,
        parent_pose=trans_mat(pos=np.array([0, 0, -0.4])),
        child_pose=trans_mat(pos=np.array([0, 0, -1.0])),
    )
)

kp = 100.0
kd = 10.0
q_start = np.array([0.5, -0.6, -1.5])
wheel_vel = np.array([5.0, 5.0, 5.0, 5.0])

normal_force = wheel_mass * 9.81
frictional_force = plane.lateral_friction * normal_force
required_torque = frictional_force * wheel_radius

for i in range(1000000):

    arm.effort = arm.inverse_dynamics(kp * (np.zeros((arm.dof,)) - arm.configuration) + kd * (np.zeros((arm.dof,)) - arm.velocity))
    chain.effort = chain.inverse_dynamics(50.0 * (wheel_vel - chain.velocity) -5.0 * chain.velocity + required_torque)
    
    if i < 2000:

        arm.velocity = np.zeros((arm.dof,))
        arm.configuration = q_start

    if i < 1000:
        chain.velocity = np.zeros((chain.dof,))
        chain.configuration = np.zeros((chain.dof,))
        chain.base_velocity = np.zeros((6,))
        chain.base_pose = np.eye(4)

    world.step(sleep=False)
