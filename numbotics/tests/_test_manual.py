import numbotics.physics as phys

import numpy as np
from scipy.spatial.transform import Rotation as R
import networkx as nx

from numbotics.math.spatial import trans_mat

import time


world = phys.World(visualize=True)
world.gravity = np.array([0, 0, -9.81])

print(world.dt)

static_block = phys.object.Cuboid(0, np.array([1.0, 1.0, 1.0]), static=True)
T = np.eye(4)
T[:3, 3] = np.array([10.0, 0.0, 0.0])
static_block.pose = T


cube = phys.object.Cube(1.0, 0.5)

T = np.eye(4)
T[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
T[:3, 3] = np.array([0, 0, 10])
cube.pose = T


manatee = phys.object.Mesh(
    10.0,
    "./numbotics/tests/models/manatee.stl",
    mesh_scale=np.array([0.01, 0.01, 0.01]),
    convex_decomposition=True,
    auto_center=True,
)
manatee.pose = trans_mat(pos=np.array([10, -0.5, 10]))


mesh = phys.object.Mesh(0.1, collision_kwargs={'filename': "./numbotics/tests/models/base_link.obj", 'auto_center': True}, linear_damping=0.0, angular_damping=0.0)
mesh.position = np.array([0, 0, 1])
# mesh.pose = T

plane = phys.object.Plane(0, np.array([0, 0, 1]), width=10, height=10, static=True)
plane.pose = trans_mat(pos=np.array([0, 0, -5]))
plane.lateral_friction = 1.0
print(plane.lateral_friction)

print(cube.pose)

sphere = phys.object.Sphere(1.0, 1.0)
sphere.position = np.array([-10, 0, 3])

# world.add_constraint(static_block, cube, phys.Constraint.PRISMATIC, joint_axis=[0, 0, 1])

cube.apply_force(np.array([0, 0, 10]), np.array([0.0, 0.0, 0]), local=True)
mesh.apply_force(np.array([10, 0, 0]), np.array([0.0, 0.0, 0]), local=True)

BT = np.eye(4)
BT[:3, 3] = np.array([0.0, 0.0, 0.0])
BT[:3, :3] = R.from_euler("y", np.pi / 2).as_matrix()
T = np.eye(4)
T[:3, 3] = np.array([1.0, 0.0, 0.0])
Tlink = np.eye(4)
Tlink[:3, 3] = np.array([0.5, 0.0, 0.0])
Tlink[:3, :3] = R.from_euler("y", -np.pi / 2.0).as_matrix()

link_list = [
    phys.DummyLink(trans_mat(), 0),
    phys.BasicLink(Tlink, mass=5.0, geometry=phys.Geometry.CYLINDER, radius=0.1, height=1.0),
    phys.BasicLink(Tlink, mass=5.0, geometry=phys.Geometry.CYLINDER, radius=0.1, height=1.0),
    phys.BasicLink(Tlink, mass=5.0, geometry=phys.Geometry.CYLINDER, radius=0.1, height=1.0),
    phys.BasicLink(Tlink, mass=5.0, geometry=phys.Geometry.CYLINDER, radius=0.1, height=1.0),
    phys.DummyLink(trans_mat(), 0.01),
]
print('TEST JOINTS--------------------------------')
joint_list = [
    phys.Joint( BT, np.array([0, 0, 1]), phys.Constraint.REVOLUTE ),
    phys.Joint( T,  np.array([0, 0, 1]), phys.Constraint.REVOLUTE ),
    phys.Joint( T,  np.array([0, 0, 1]), phys.Constraint.REVOLUTE ),
    phys.Joint( T,  np.array([0, 0, 1]), phys.Constraint.REVOLUTE ),
    phys.Joint( T,  np.array([0, 0, 0]), phys.Constraint.FIXED    ),
]
print('--------------------------------')

chain = phys.SerialChain(
    link_list,
    joint_list,
)

# chain.base_position = np.array([2, 2, 0])

# world.dt = 0.01

chain.joint_damping = np.array([1.0, 1.0, 1.0, 1.0])
chain.joint_limits = np.array(
    [
        [-100.0, 100.0],
        [-100.0, 100.0],
        [-100.0, 100.0],
        [-100.0, 100.0],
    ]
)
chain.joint_effort_limits = np.array([350.0, 350.0, 350.0, 350.0])

chain.joint_damping[:2] = 1e-5
chain.joint_limits[:2] = np.array([[-50.0, 50.0], [-50.0, 50.0]])
chain.joint_effort_limits[:4] = 40000000.0

chain.base_position = np.array([0, 1.5, 0])

block_weld = phys.Joint(
    np.eye(4),
    np.array([0, 0, 1]),
    phys.Constraint.FIXED,
    parent_pose=trans_mat(pos=np.array([0, 0, 2])),
    child_pose=np.eye(4),
)

def test_stack_del():
    static_block = phys.object.Cuboid(0, np.array([1.0, 1.0, 1.0]), static=True)
    T = np.eye(4)
    T[:3, 3] = np.array([-10.0, 0.0, 0.0])
    static_block.pose = T
    world.step()
    time.sleep(1)

def test_chain_del():
    chain = phys.SerialChain(
        link_list,
        joint_list,
    )
    chain.base_position = np.array([0, 2, 0])
    world.step()
    time.sleep(1)

print(chain.mass_matrix)
print(chain.coriolis_centrifugal_vector)
print(chain.gravity_vector)

test_stack_del()
test_chain_del()

start_img_t = time.time()
image = world.depth_image(780, 520, trans_mat(pos=np.array([-10, 0, 10])))
img_end_t = time.time()
print(f"Image time: {img_end_t - start_img_t} seconds")

chain.configuration = np.array([1.8, 1.0, -2.0, 0.3])

chain.velocity = np.array([0.4, -0.6, 0.2, 1.4])


fin_diff_start_t = time.time()
res1 = chain.coriolis_centrifugal_matrix @ chain.velocity
fin_diff_end_t = time.time()
print(f"Finite difference for coriolis matrix: {fin_diff_end_t - fin_diff_start_t} seconds")

direct_start_t = time.time()
res2 = chain.coriolis_centrifugal_vector
direct_end_t = time.time()
print(f"Direct computation of coriolis vector: {direct_end_t - direct_start_t} seconds")

print(f'finite diff error: {np.linalg.norm(res1 - res2)}')


ik_time_start = time.time()
q_des = chain.inverse_kinematics(np.array([0, 1, 2]), seed=np.array([0, 0, 0, 0]))
ik_time_end = time.time()
print(f"IK time: {ik_time_end - ik_time_start} seconds")

print(chain.jacobian().shape)

print(len(distances := chain.distance_to(static_block)))
print(distances[0].distance)

del distances


T_des = trans_mat(pos=np.array([10, 10, 0]))
w_vel = 4.0
def control():
    T = cube.pose
    p_err = T_des[:3, 3] - T[:3, 3]
    R_err = -0.5 * np.cross(T_des[:3, 2], T[:3, 2])
    u = np.concatenate([p_err, R_err])# np.sum(R_err,axis=0)])
    
    vel_err = np.array([0.0, 0.0, 0.0, *(w_vel * T[:3, 2])]) - cube.velocity
    
    u_g = np.zeros((6,))
    u_g[2] = cube.mass * 9.81
    return (np.diag([10.0, 10.0, 10.0, 2.0, 2.0, 2.0]) @ u) + (np.diag([20.0, 20.0, 20.0, 4.0, 4.0, 4.0]) @ vel_err) + u_g

cube_actuator = phys.Actuator(cube, control)

kp = 10.0
kd = 20.0
qd_des = np.array([0.0, 0.0, 0.0, 0.0])


def control_chain():
    def skew_to_vec(S):
        return np.array([
            S[2, 1],
            S[0, 2],
            S[1, 0]
        ])
    T_des = trans_mat(pos=np.array([0, 0, 5]))
    T = chain.base_pose
    p_err = T_des[:3, 3] - T[:3, 3]
    R_err = T_des[:3, :3] @ T[:3, :3].T
    R_err = -skew_to_vec(0.5 * (R_err - R_err.T))
    u = np.concatenate([p_err, 30.0 * R_err])
    
    vel_err = - np.array([10.0, 10.0, 10.0, 1e-1, 1e-1, 1e-1]) * chain.base_velocity
 
    return 20.0 * u + 7.0 * vel_err

def chain_u():
    return chain.inverse_dynamics(
        np.hstack((control_chain(), (kp * (q_des - chain.configuration)) + (kd * (qd_des - chain.velocity))))
    )

# chain_actuator = phys.Actuator(chain._links[0], lambda : chain_u()[:6])

chain.configuration = np.array([0.0, 0.0, 0.0, 0.0])
print('init duplicate chain')
duplicate_chain = phys.helpers._chain_from_pyb(chain._pyb_id)
print('finished init duplicate chain')
duplicate_chain.configuration = np.ones((4,))


chain.base_position = np.array([-1, 0, 0])

for _ in range(10000000):
    
    world.step(sleep=True)

    chain.effort = chain.inverse_dynamics((kp * (q_des - chain.configuration)) + (kd * (qd_des - chain.velocity)))
    # print('duplicate chain')
    duplicate_chain.effort = duplicate_chain.inverse_dynamics((kp * (q_des - duplicate_chain.configuration)) + (kd * (qd_des - duplicate_chain.velocity)))
    
    if _ == 1000:
        print(chain.forward_kinematics())

        for contact in plane.contacts:
            # Unsafe delete. This deletes the underlying Pybullet data and visualization
            # data, but it does not clean up local references, i.e. mesh = Mesh(...)
            # will still hold a reference to the Mesh object with bad Pybullet ID.
            world.unregister(contact.target)

        # Safe delete. Removes single local reference to static_block, and deconstructor
        # will clean up the Pybullet and visualization data.
        del static_block


print(cube.pose)
print(cube.velocity)
