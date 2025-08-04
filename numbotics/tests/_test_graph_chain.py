import numbotics.physics as phys
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import networkx as nx
from numbotics.math.spatial import trans_mat


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
body = phys.BasicLink(body_transform, mass=500.0, geometry=phys.Geometry.CUBOID, half_extents=np.array([1.0, 0.5, 0.2]), static=False)
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

chain.base_pose = trans_mat(
    pos=np.array([0, 0, 1.0]),
)

# Set up a simple control loop
kp = 50.0  # Proportional gain
kd = 5.0   # Derivative gain

# Target positions for each joint (in radians)
target_positions = np.zeros(4)

# FWD
target_velocities = np.array([5.0, 5.0, 5.0, 5.0])

# Run the simulation
for _ in range(1000000):
    # Calculate control torques using PD control
    position_error = np.zeros_like(target_positions) # - chain.configuration
    velocity_error = target_velocities - chain.velocity
    
    # PD control law
    control_torques = (kp * velocity_error) / np.array([0.5 * wheel_mass * wheel_radius**2 for i in range(chain.dof)]) # - kd * chain.effort
    # Apply the control torques
    if _ > 500:
        chain.effort = control_torques
    
    # Step the simulation
    world.step(sleep=True)
    # time.sleep(0.01)  # Slow down the simulation for visualization
    
    # Every 200 steps, change the target positions
    if _ % 20 == 0 and _ > 0:
        print('--------------------------------')
        print(chain.configuration)
        print(chain.effort)
        print(control_torques)
        print(velocity_error)
        print(chain.velocity)
        # target_positions = np.random.uniform(-1.0, 1.0, 4)
        # print(f"New target positions: {target_positions}")

# Print final configuration
print("Final joint configuration:", chain.configuration)
print("Final joint velocities:", chain.velocity)