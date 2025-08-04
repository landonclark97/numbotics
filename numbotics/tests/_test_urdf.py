import numbotics.physics as phys
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import networkx as nx
from numbotics.math.spatial import trans_mat



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()




world = phys.World(visualize=args.vis)
# world.dt = 1.0 / 1000.0
world.gravity = np.array([0, 0, 0])


plane = phys.object.Plane(0, np.array([0, 0, 1]), width=10, height=10, static=True)

chain = phys.GraphChain.from_urdf('./numbotics/tests/models/kortex/kinova_no_obs.urdf')
# obs = phys.helpers._chain_from_urdf('./numbotics/tests/models/objects_kinova.urdf')

chain.configuration = np.random.uniform(chain.joint_limits[:, 0] + 1e-1, chain.joint_limits[:, 1] - 1e-1, (chain.dof,))
chain.configuration[0] = 2.0
print(type(chain.configuration))
print(type(chain.configuration))

print(f'DOF: {chain.dof}')

dt = world.dt
K_d = np.diag(np.ones_like(chain.configuration) * 10.0)
K_p = np.diag(np.ones_like(chain.configuration) * 20.0)

# Run the simulation
for _ in range(1000000):

    world.step(sleep=True)

    # https://faculty.cc.gatech.edu/~turk/my_papers/stable_pd.pdf
    C = chain.noninertial_dynamics
    M = chain.mass_matrix
    D = chain.joint_damping * chain.velocity

    q_err = -chain.configuration
    qd_err = -chain.velocity

    p_term = K_p @ (q_err)# - (chain.velocity * dt))
    d_term = K_d @ (qd_err)

    qddot = np.linalg.solve(a=(M + K_d * dt), b=(-C - D + p_term + d_term))

    chain.effort = p_term + d_term - ((K_d @ qddot) * dt)
