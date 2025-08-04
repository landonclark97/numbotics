import numbotics.physics as phys
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import networkx as nx
from numbotics.math.spatial import trans_mat

from numbotics.robots import Arm

import time


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vis', action='store_true')
args = parser.parse_args()


world = phys.World(visualize=args.vis)
world.gravity = np.array([0, 0, -9.81])


plane = phys.object.Plane(0, np.array([0, 0, 1]), width=10, height=10, static=True)

chain = phys.GraphChain.from_urdf('./numbotics/tests/models/kortex/kinova_no_obs.urdf')
# chain = phys.GraphChain.from_urdf('./numbotics/tests/models/kortex/kinova_cyl.urdf')
# obs = phys.helpers._chain_from_urdf('./numbotics/tests/models/objects_kinova.urdf')

chain.configuration = np.random.uniform(chain.joint_limits[:, 0] + 1e-1, chain.joint_limits[:, 1] - 1e-1, (chain.dof,))
chain.configuration = np.zeros((chain.dof,))
print(type(chain.configuration))
print(type(chain.configuration))

print(f'DOF: {chain.dof}')

world.step()

arm = Arm(chain)
start = time.time()
arm.base_pose
end = time.time()
print(f'Time taken BP1: {end - start}')

start = time.time()
arm.base_pose
end = time.time()
print(f'Time taken BP2: {end - start}')

for _ in range(10):
    q = np.random.uniform(arm._chain.joint_limits[:, 0] + 1e-1, arm._chain.joint_limits[:, 1] - 1e-1, (arm._chain.dof,))
    arm._chain.configuration = q
    for i, link in enumerate(chain._links):
        if link.name == 'tool_frame':
            tool_frame_idx = i
            T = arm.forward_kinematics(q, 'tool_frame')
            print(np.linalg.norm(T - link.pose))

Q = np.random.uniform(arm._chain.joint_limits[:, 0] + 1e-1, arm._chain.joint_limits[:, 1] - 1e-1, (1000, arm._chain.dof,))

for _ in range(15):
    start = time.time()
    T = arm.forward_kinematics(Q, 'tool_frame')
    end = time.time()
    print(f'Time taken FK: {end - start}')

start = time.time()
arm._chain.base_pose
end = time.time()
print(f'Time taken: {end - start}')

arm._chain.configuration = Q[0]
J_pyb = arm._chain.jacobian(tool_frame_idx)
J_numb = arm.jacobian(Q[0], 'tool_frame')
print(np.linalg.norm(J_pyb - J_numb))

start = time.time()
J_numb = arm.jacobian(Q, 'tool_frame')
end = time.time()
print(f'Time taken: {end - start}')

start = time.time()
J_numb = arm.jacobian(Q, 'tool_frame')
end = time.time()
print(f'Time taken: {end - start}')

Trand = trans_mat(pos=np.random.uniform(-0.2, 0.2, 3), orn=R.random().as_matrix())
start = time.time()
s, q = arm.inverse_kinematics(Trand, Q, 'tool_frame')
end = time.time()
print(f'Time taken our IK: {end - start}')
print(f'Success rate: {100 * np.mean(s)}%')

start = time.time()
q_pyb = arm._chain.inverse_kinematics(Trand[:3, 3], Trand[:3, :3], Q[0], tool_frame_idx)
end = time.time()
print(f'Time taken pyb IK: {end - start}')

start = time.time()
arm.self_collisions(Q[0])
end = time.time()
print(f'Time taken self collision: {end - start}')

print(len(arm.collision_pairs()))
print(len(arm.self_collision_pairs()))
print(len(arm.self_collisions(Q[0])))
print(len(arm._chain._links) * len(arm._chain._links))

quit()

d_safe = 0.035
k_d = 5.0
for _ in range(1000000):
    configuration = arm._chain.configuration
    C = arm.self_collisions(configuration)
    J = np.zeros((0, arm._chain.dof))
    for c in C:
        if c.distance < d_safe:

            weight = 1.0 - (c.distance/d_safe)**2
            if weight < 0.0:
                weight = 1.0
            
            local_pose_a = (np.linalg.inv(arm.forward_kinematics(configuration, c.subject.name)) @ np.array([*c.position_on_subject, 1]))[:-1]
            delta_a = weight * c.normal_target_to_subject @ arm.jacobian(configuration, c.subject.name, local_pose=trans_mat(pos=local_pose_a))[:3]

            local_pose_b = (np.linalg.inv(arm.forward_kinematics(configuration, c.target.name)) @ np.array([*c.position_on_target, 1]))[:-1]
            delta_b = weight * c.normal_target_to_subject @ arm.jacobian(configuration, c.target.name, local_pose=trans_mat(pos=local_pose_b))[:3]

            dq = np.concatenate((np.atleast_2d(delta_a), np.atleast_2d(-delta_b)), axis=0)
            J = np.concatenate((J, dq), axis=0)

    if J.shape[0] > 0:
        H = J.T @ J
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        idx_max = np.argmax(eigenvalues)
        q_max = eigenvectors[:, idx_max] * np.sqrt(eigenvalues[idx_max].real)
    else:
        q_max = np.zeros((arm._chain.dof,))

    print(f'velocity magnitude: {np.linalg.norm(q_max)}')

    arm._chain.effort = arm._chain.inverse_dynamics((q_max - arm._chain.velocity) * k_d)

    world.step()