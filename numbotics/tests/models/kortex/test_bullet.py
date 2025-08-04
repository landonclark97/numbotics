import numpy as np

import pybullet as p

from scipy.spatial.transform import Rotation as R


p.connect(p.GUI)

# o = p.loadURDF('obstacles.urdf')

# r_other = p.loadURDF('kinova_no_obs.urdf')
r = p.loadURDF('kinova_cyl.urdf')

# p.resetJointState(r, 2, np.deg2rad(285.5))
# p.resetJointState(r, 3, np.deg2rad(315.0))
# p.resetJointState(r, 4, np.deg2rad(122.9))
# p.resetJointState(r, 5, np.deg2rad(89.3))
# p.resetJointState(r, 6, np.deg2rad(288.1))
# p.resetJointState(r, 7, np.deg2rad(43.7))
# p.resetJointState(r, 8, np.deg2rad(31.6))

# p.resetJointState(r_other, 1, np.deg2rad(285.5))
# p.resetJointState(r_other, 2, np.deg2rad(315.0))
# p.resetJointState(r_other, 3, np.deg2rad(122.9))
# p.resetJointState(r_other, 4, np.deg2rad(89.3))
# p.resetJointState(r_other, 5, np.deg2rad(288.1))
# p.resetJointState(r_other, 6, np.deg2rad(43.7))
# p.resetJointState(r_other, 7, np.deg2rad(31.6))


# c = p.getClosestPoints(r, r, distance=0)
# print(c)

# for i in range(p.getNumJoints(r)):
#     print(p.getJointInfo(r,i))


for joint_index in range(p.getNumJoints(r)):
    joint_info = p.getJointInfo(r, joint_index)
    link_name = joint_info[12].decode('UTF-8')

    link_state = p.getLinkState(r, joint_index)
    link_pos = list(link_state[0])
    link_quat = link_state[1]

    coll_shape_data = p.getCollisionShapeData(r, joint_index)
    print(coll_shape_data)
    try:
        coll_com_pos = coll_shape_data[0][5]
        coll_quat = coll_shape_data[0][6]
    except:
        continue

    link_pos = list(p.multiplyTransforms(link_pos, link_quat, coll_com_pos, coll_quat)[0])
    com_link_pos = link_pos[:]
    link_pos[1] = link_pos[1] - 0.5
    link_pos[2] = link_pos[2] + 0.5

    # p.addUserDebugText(link_name, link_pos, textColorRGB=[0, 0, 0], textSize=1.5)
    # p.addUserDebugLine(com_link_pos, link_pos, lineColorRGB=[1, 0, 0], lineWidth=2.0)

import time
while True:
    time.sleep(0.4)
