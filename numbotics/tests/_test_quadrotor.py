import numbotics.physics as phys
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import networkx as nx
from numbotics.math.spatial import trans_mat
from numbotics.utils import Shape


# Create a visualization world
world = phys.World(visualize=True)
world.gravity = np.array([0, 0, -9.81])

plane = phys.Plane(
    0, 
    np.array([0, 0, 1]), 
    static=True
)
plane.lateral_friction = 0.9
plane.rolling_friction = 1e-4
plane.spinning_friction = 1e-4

plane.position = np.array([0, 0, -4])

# Create a directed graph for our car structure
G = nx.DiGraph()

# Create transformation matrices for the car body and wheels
body_transform = np.eye(4)

# Create the car body (root node)
body = phys.BasicLink(1.0, body_transform, Shape.CUBOID, half_extents=np.array([0.5, 0.5, 0.1]))
G.add_node(0, link=body)

# Create rotors
rotor_radius = 0.3
rotor_width = 0.05
rotor_mass = 0.1

# Define rotor positions relative to the body
rotor_positions = [
    np.array([0.8, 0.8, 0.2]),  # Front right
    np.array([0.8, -0.8, 0.2]), # Front left
    np.array([-0.8, 0.8, 0.2]), # Rear right
    np.array([-0.8, -0.8, 0.2]) # Rear left
]

# Create wheels and add them to the graph
for i, pos in enumerate(rotor_positions, start=1):
    rotor = phys.BasicLink(rotor_mass, np.eye(4), Shape.CYLINDER, radius=rotor_radius, height=rotor_width)
    G.add_node(i, link=rotor)

    # Create revolute joints for the wheels
    joint = phys.Joint(trans_mat(pos=pos), np.array([0, 0, 1]), phys.Constraint.FIXED)
    G.add_edge(0, i, joint=joint)

# Create the GraphChain
chain = phys.GraphChain(G, static_base=False)


chain.base_pose = trans_mat(
    pos=np.array([0, 0, 5.0]),
)

def control_law():
    return np.array([0, 0, 1.4 * 9.81, 0, 0, 0])


actuator = phys.Actuator(chain._links[0], control_law)
chain.configuration

# Run the simulation
for _ in range(1000000):

    world.step(sleep=True)
