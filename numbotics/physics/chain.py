import copy

import numpy as np
from scipy.spatial.transform import Rotation as _R
from scipy.differentiate import jacobian
import networkx as nx

import numbotics.physics as physics
from numbotics.physics import pyb
from numbotics.graphics import VisualShape
from numbotics.math import trans_mat
from numbotics.utils import parse_shape_kwargs, Shape, logger

from .collision import CollisionShape
from .constraint import Joint, Constraint
from .helpers import (
    _ConfigurationArray,
    _VelocityArray,
    _EffortArray,
    _JointDampingArray,
    _JointLimitsArray,
    _JointEffortLimitsArray,
    _chain_from_urdf,
    SimpleBulletClient,
)



class Link:

    def __init__(
            self,
            mass: float,
            offset: np.ndarray,
            collision_shape: CollisionShape | None = None,
            visual_shape: VisualShape | None = None,
            **kwargs,
        ):

        if mass < 0:
            raise ValueError("Link mass must be positive")

        self._offset = offset
        self._mass = mass

        self._collision_shape = collision_shape if collision_shape is not None else physics.CollisionShape(Shape.EMPTY)
        self._visual_shape = visual_shape if visual_shape is not None else VisualShape(Shape.EMPTY)
            
        if self._visual_shape.shape == Shape.EMPTY and self._collision_shape.shape != Shape.EMPTY:
            visual_shape_type = self._collision_shape.shape
            if visual_shape_type is None:
                raise ValueError(f"Invalid shape type: {self._collision_shape.shape}")
            self._visual_shape = VisualShape(
                visual_shape_type,
                **self._collision_shape._shape_info,
            )

        self._body_id = None
        self._index = None

        self._world_name = None
        self._pyb_client = None

        self._inertia_diagonal = kwargs.pop('inertia_diagonal', None)
        self._name = kwargs.pop('name', None)

        for key in list(kwargs.keys()):
            if hasattr(self, f"{key}"):
                setattr(self, f"{key}", kwargs.pop(key))
        
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")


    def __str__(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return f"{self._world_name}:entity_id_{self._body_id}:sub_id_{self._index}"


    def __eq__(self, other):
        return str(self) == str(other)


    def __hash__(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return hash(self.name)


    @property
    def world(self):
        return physics.get_world(name=self.world_name)


    @property
    def body_id(self):
        return self._body_id
    

    @body_id.setter
    def body_id(self, body_id: int):
        self._body_id = body_id


    @property
    def index(self):
        return self._index
    

    @index.setter
    def index(self, index: int):
        self._index = index
        if self._name is None:
            self._name = f'link_{self._index}'


    @property
    def world_name(self):
        return self._world_name


    @world_name.setter
    def world_name(self, world_name: str):
        self._world_name = world_name


    @property
    def name(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        body_name = self.world._pyb_entity_map[self._body_id]._name
        return f'{self._world_name}:{body_name}:{self._name}'


    @property
    def mass(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return self._pyb_client.getDynamicsInfo(self._body_id, self._index)[0]
    

    @mass.setter
    def mass(self, mass: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")        
        self._mass = mass
        self._pyb_client.changeDynamics(self._body_id, self._index, mass=mass)

    
    @property
    def inertia_diagonal(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return np.array(self._pyb_client.getDynamicsInfo(self._body_id, self._index)[2])
    

    @inertia_diagonal.setter
    def inertia_diagonal(self, inertia_diagonal: np.ndarray):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.changeDynamics(self._body_id, self._index, localInertiaDiagonal=inertia_diagonal.tolist())


    @property
    def linear_damping(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        if not hasattr(self, '_linear_damping'):
            self._linear_damping = 0.04
        return self._linear_damping


    @linear_damping.setter
    def linear_damping(self, linear_damping: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._linear_damping = linear_damping
        self._pyb_client.changeDynamics(self._body_id, self._index, linearDamping=linear_damping)


    @property
    def angular_damping(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        if not hasattr(self, '_angular_damping'):
            self._angular_damping = 0.04
        return self._angular_damping


    @angular_damping.setter
    def angular_damping(self, angular_damping: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._angular_damping = angular_damping
        self._pyb_client.changeDynamics(self._body_id, self._index, angularDamping=angular_damping)


    @property
    def restitution(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return self._pyb_client.getDynamicsInfo(self._body_id, self._index)[5]
    

    @restitution.setter
    def restitution(self, restitution: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.changeDynamics(self._body_id, self._index, restitution=restitution)


    @property
    def lateral_friction(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return self._pyb_client.getDynamicsInfo(self._body_id, self._index)[1]


    @lateral_friction.setter
    def lateral_friction(self, lateral_friction: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.changeDynamics(self._body_id, self._index, lateralFriction=lateral_friction)


    @property
    def rolling_friction(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return self._pyb_client.getDynamicsInfo(self._body_id, self._index)[6]


    @rolling_friction.setter
    def rolling_friction(self, rolling_friction: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.changeDynamics(self._body_id, self._index, rollingFriction=rolling_friction)


    @property
    def spinning_friction(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return self._pyb_client.getDynamicsInfo(self._body_id, self._index)[7]


    @spinning_friction.setter
    def spinning_friction(self, spinning_friction: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.changeDynamics(self._body_id, self._index, spinningFriction=spinning_friction)


    @property
    def contact_damping(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return self._pyb_client.getDynamicsInfo(self._body_id, self._index)[8]


    @contact_damping.setter
    def contact_damping(self, contact_damping: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.changeDynamics(self._body_id, self._index, contactDamping=contact_damping)


    @property
    def contact_stiffness(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        return self._pyb_client.getDynamicsInfo(self._body_id, self._index)[9]


    @contact_stiffness.setter
    def contact_stiffness(self, contact_stiffness: float):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.changeDynamics(self._body_id, self._index, contactStiffness=contact_stiffness)


    @property
    def pose(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        if self._index == -1:
            p, q = self._pyb_client.getBasePositionAndOrientation(self._body_id)
        else:
            state = self._pyb_client.getLinkState(
                self._body_id, self._index, computeForwardKinematics=False
            )
            p, q = state[4], state[5]
        return trans_mat(pos=np.array(p), orn=_R.from_quat(q).as_matrix())
    

    def apply_wrench(self, wrench: np.ndarray, position: np.ndarray = np.zeros((3,)), local: bool = False):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.applyExternalForce(
            self._body_id, self._index, wrench[:3].tolist(), position.tolist(), self._pyb_client.LINK_FRAME if local else self._pyb_client.WORLD_FRAME
        )
        self._pyb_client.applyExternalTorque(
            self._body_id, self._index, wrench[3:].tolist(), self._pyb_client.LINK_FRAME if local else self._pyb_client.WORLD_FRAME
        )


    def apply_force(self, force: np.ndarray, position: np.ndarray = np.zeros((3,)), local: bool = False):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.applyExternalForce(
            self._body_id, self._index, force.tolist(), position.tolist(), self._pyb_client.LINK_FRAME if local else self._pyb_client.WORLD_FRAME
        )


    def apply_torque(self, torque: np.ndarray, local: bool = False):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        self._pyb_client.applyExternalTorque(self._body_id, self._index, torque.tolist(), self._pyb_client.LINK_FRAME if local else self._pyb_client.WORLD_FRAME)


    @property
    def contacts(self):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        contact_list = []
        contacts = self._pyb_client.getContactPoints(self._body_id, self._index)
        for contact in contacts:
            if (target := self.world._pyb_entity_map.get(contact[2])) is None:
                raise ValueError(f"Target {contact[2]} is not a valid object in the world")
            if isinstance(target, Chain):
                target = target._links_from_indices[contact[4]]
            
            contact_list.append(
                physics.Contact(
                    subject=self,
                    target=target,
                    position_on_subject=np.array(contact[5]),
                    position_on_target=np.array(contact[6]),
                    contact_normal_target_to_subject=np.array(contact[7]),
                    contact_distance=contact[8],
                    normal_force=contact[9],
                    lateral_friction_a=contact[10],
                    lateral_friction_a_dir=np.array(contact[11]),
                    lateral_friction_b=contact[12],
                    lateral_friction_b_dir=np.array(contact[13]),
                )
            )
        return contact_list
    

    def distance_to(self, target: 'physics.PhysicsObject | Link | Chain', max_distance: float = np.inf):
        if self._body_id is None or self._index is None or self._world_name is None or self._pyb_client is None:
            raise ValueError("Link is not registered in the world")
        proximity_list = []
        distances = self._pyb_client.getClosestPoints(
            self._body_id, 
            target._body_id if isinstance(target, Link) else target._pyb_id, 
            max_distance, 
            linkIndexA=self._index,
            **({'linkIndexB': target._index} if isinstance(target, Link) else {}),
        )
        for distance in distances:
            if (target := self.world._pyb_entity_map.get(distance[2])) is None:
                raise ValueError(f"Target {distance[2]} is not a valid object in the world")
            if isinstance(target, Chain):
                target = target._links_from_indices[distance[4]]
            
            proximity_list.append(
                physics.Proximity(
                    subject=self,
                    target=target,
                    position_on_subject=np.array(distance[5]),
                    position_on_target=np.array(distance[6]),
                    normal_target_to_subject=np.array(distance[7]),
                    distance=distance[8],
                )
            )
        return proximity_list



class DummyLink(Link):
    def __init__(
            self,
            mass: float,
            offset: np.ndarray,
            **kwargs,
        ):
        super().__init__(mass, offset, None, None, **kwargs)



class BasicLink(Link):
    def __init__(
            self,
            mass: float,
            offset: np.ndarray,
            shape: Shape,
            **kwargs,
        ):
        kwargs, shape_info = parse_shape_kwargs(kwargs)
        super().__init__(mass, offset, physics.CollisionShape(shape, **shape_info), None, **kwargs)



def _create_multibody(
    world: 'physics.World',
    static_base: bool,
    base_link: Link,
    links: list[Link],
    joints: list[Joint],
    link_masses: list[float],
    link_positions: list[list[float]],
    link_orientations: list[list[float]],
    link_parent_indices: list[int],
    link_inertial_positions: list[list[float]],
    link_inertial_orientations: list[list[float]],
    joint_types: list[int],
    joint_axes: list[list[float]],
):
    
    base_link._collision_shape.register(pyb_client=world._pyb_client)
    for link in links:
        link._collision_shape.register(pyb_client=world._pyb_client)

    base_id = world._pyb_client.createMultiBody(
        baseMass=0 if static_base else base_link._mass,
        baseCollisionShapeIndex=base_link._collision_shape.col_id,
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0],
        baseOrientation=[0, 0, 0, 1],
        baseInertialFramePosition=base_link._offset[:3, 3].tolist(),
        baseInertialFrameOrientation=_R.from_matrix(base_link._offset[:3, :3]).as_quat().tolist(),
        linkMasses=link_masses,
        linkCollisionShapeIndices=[link._collision_shape.col_id for link in links],
        linkVisualShapeIndices=[-1 for _ in links],
        linkPositions=link_positions,
        linkOrientations=link_orientations,
        linkParentIndices=link_parent_indices,
        linkInertialFramePositions=link_inertial_positions,
        linkInertialFrameOrientations=link_inertial_orientations,
        linkJointTypes=joint_types,
        linkJointAxis=joint_axes,
    )

    base_link.index = -1
    joint_to_link_map = {}
    for i in range(len(joints)):
        joint_to_link_map[i] = int(world._pyb_client.getJointInfo(base_id, i)[1].decode('utf-8').replace('joint', '')) - 1
    link_to_joint_map = {v: k for k, v in joint_to_link_map.items()}
    for i, link in enumerate(links):
        link.index = link_to_joint_map[i]

    # Enable torque control
    for i in range(len(joints)):
        world._pyb_client.setJointMotorControl2(
            base_id, i, controlMode=world._pyb_client.VELOCITY_CONTROL, force=0
        )

    return base_id



class Chain:

    def __init__(
        self,
        pyb_id: int,
        links: list[Link],
        joints: list[Joint],
        static_base: bool = True,
        **kwargs,
    ):
        if not hasattr(self, '_world_name'):
            self._world_name = kwargs.pop('world_name', None)
        world = physics.get_world(name=self._world_name)
        if not hasattr(self, '_pyb_client'):
            self._pyb_client = SimpleBulletClient(world._pyb_client._client)

        self._pyb_id = pyb_id
        self._static = False
        self._static_base = static_base if links[0]._mass > 0 else True
        self._links = links
        self._joints = joints

        for link in self._links:
            link._body_id = self._pyb_id
            link._world_name = self._world_name
            link._pyb_client = self._pyb_client
            if isinstance(link, DummyLink):
                # Set inertia matrix to unit sphere
                i_diag = (2.0 / 5.0) * link._mass
                self._pyb_client.changeDynamics(
                    self._pyb_id,
                    link._index if link._index is not None else -1,
                    # Set inertia matrix to something to improve stability
                    # TODO: We need to figure out what this should actually be
                    localInertiaDiagonal=[i_diag, i_diag, i_diag],
                )

        self._name = kwargs.pop('name', f'chain_{self._pyb_id}')

        for key in list(kwargs.keys()):
            setattr(self, f"_{key}", kwargs.pop(key))
        
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        for link in self._links:
            if link._inertia_diagonal is not None:
                link.inertia_diagonal = link._inertia_diagonal

        self._links_from_indices = {link._index: link for link in self._links}

        self.__floating_base_indices = list(range(6)) + list(range(7, 7 + self.dof))
    
        self.__single_dof_indices = []
        self.__multi_dof_indices = []
        self.__single_dof_indices_pyb = []
        self.__multi_dof_indices_pyb = []
        self.__joint_to_index = {}
        j_idx = 0
        for joint, link in zip(self._joints, self._links[1:]):
            if joint.type == Constraint.PRISMATIC or joint.type == Constraint.REVOLUTE:
                self.__single_dof_indices.append(j_idx)
                self.__single_dof_indices_pyb.append(link._index)
                self.__joint_to_index[joint] = j_idx
                j_idx += 1
            elif joint.type == Constraint.SPHERICAL:
                self.__multi_dof_indices.extend([j_idx, j_idx + 1, j_idx + 2])
                self.__multi_dof_indices_pyb.extend([link._index])
                self.__joint_to_index[joint] = [j_idx, j_idx + 1, j_idx + 2]
                j_idx += 3

        self.__nonfixed_indices_pyb = [link._index for joint, link in zip(self._joints, self._links) if joint.type != Constraint.FIXED]

        self.joint_damping = np.array([joint.damping for joint in self._joints if joint.type != Constraint.FIXED])
        self.joint_limits = np.array([[joint.lower_limit, joint.upper_limit] for joint in self._joints if joint.type != Constraint.FIXED])
        self.joint_effort_limits = np.array([joint.max_effort for joint in self._joints if joint.type != Constraint.FIXED])

        world.register(self)


    def __del__(self):
        if self._pyb_id is not None:
            if self._world_name is not None:
                try:
                    self._pyb_client.removeBody(self._pyb_id)
                except pyb.error:
                    pass
                try:
                    physics.get_world(name=self._world_name).unregister(self)
                except KeyError:
                    pass
            self._world_name = None
            self._pyb_id = None


    def __str__(self):
        return f"{self._world_name}:entity_id_{self._pyb_id}"


    def __eq__(self, other):
        return str(self) == str(other)


    def __hash__(self):
        return hash(self.name)
        

    @classmethod
    def inferred_attrs(cls):
        return ['pyb_id', 'pyb_client']
    

    @property
    def name(self):
        if self._name is not None:
            return f'{self._world_name}:{self._name}'
        else:
            return f'{self._world_name}:chain_{self._pyb_id}'
    

    @property
    def world(self):
        return physics.get_world(name=self._world_name)


    @property
    def dof(self):
        if not hasattr(self, '_dof'):
            self._dof = sum(joint.dof for joint in self._joints)
        return self._dof


    @property
    def poses(self):
        poses = np.zeros((len(self._links), 4, 4))
        p, q = self._pyb_client.getBasePositionAndOrientation(self._pyb_id)
        poses[0] = trans_mat(pos=np.array(p), orn=_R.from_quat(q).as_matrix())
        states = self._pyb_client.getLinkStates(self._pyb_id, [link.index for link in self._links[1:]])
        for i, state in enumerate(states):
            poses[i+1] = trans_mat(
                pos=np.array(state[4]),
                orn=_R.from_quat(state[5]).as_matrix()
            )
        return poses
    

    @property
    def joint_damping(self):
        if not hasattr(self, '_joint_damping'):
            self._joint_damping = np.zeros((self.dof,))
        return _JointDampingArray(self, self._joint_damping)


    @joint_damping.setter
    def joint_damping(self, damping: np.ndarray):
        if damping.shape != (self.dof,):
            raise ValueError(f"Damping must be a {self.dof} array")
        for i in range(self.dof):
            self._pyb_client.changeDynamics(self._pyb_id, self.__nonfixed_indices_pyb[i], jointDamping=float(damping[i]))
        self._joint_damping = np.copy(damping)


    @property
    def joint_limits(self):
        if not hasattr(self, '_joint_limits'):
            self._joint_limits = np.zeros((self.dof, 2))
        return _JointLimitsArray(self, self._joint_limits)
    

    @joint_limits.setter
    def joint_limits(self, limits: np.ndarray):
        if limits.shape == (self.dof,) and self.dof == 0:
            return
        if limits.shape != (self.dof, 2):
            raise ValueError(f"Limits must be a {self.dof}x2 array")
        for i in range(self.dof):
            self._pyb_client.changeDynamics(self._pyb_id, self.__nonfixed_indices_pyb[i], jointLowerLimit=float(limits[i, 0]), jointUpperLimit=float(limits[i, 1]))
        self._joint_limits = np.copy(limits)


    @property
    def joint_effort_limits(self):
        if not hasattr(self, '_joint_effort_limits'):
            self._joint_effort_limits = np.zeros((self.dof,))
        return _JointEffortLimitsArray(self, self._joint_effort_limits)
    

    @joint_effort_limits.setter
    def joint_effort_limits(self, limits: np.ndarray):
        if limits.shape != (self.dof,):
            raise ValueError(f"Limits must be a {self.dof} array")
        self._joint_effort_limits = np.copy(limits)


    @property
    def configuration(self):
        config = np.zeros((self.dof,))
        if self.__single_dof_indices:
            for s_ind, s_pyb_ind in zip(self.__single_dof_indices, self.__single_dof_indices_pyb):
                config[s_ind] = self._pyb_client.getJointState(self._pyb_id, s_pyb_ind)[0]
        if self.__multi_dof_indices:
            m_inds = np.array(self.__multi_dof_indices).reshape(-1, 3)
            for m_ind, m_pyb_ind in zip(m_inds, self.__multi_dof_indices_pyb):
                config[m_ind] = _R.from_quat(self._pyb_client.getJointStateMultiDof(self._pyb_id, m_pyb_ind)[0]).as_rotvec()
        return _ConfigurationArray(self, config)


    @configuration.setter
    def configuration(self, config: np.ndarray):
        if isinstance(config, (list, np.ndarray)):
            if len(config) != self.dof:
                raise ValueError(
                    f"Configuration length {len(config)} doesn't match number of joints {self.dof}"
                )
            if self.__single_dof_indices:
                for s_ind, s_pyb_ind in zip(self.__single_dof_indices, self.__single_dof_indices_pyb):
                    self._pyb_client.resetJointState(self._pyb_id, s_pyb_ind, float(config[s_ind]))
            if self.__multi_dof_indices:
                m_inds = np.array(self.__multi_dof_indices).reshape(-1, 3)
                for m_ind, m_pyb_ind in zip(m_inds, self.__multi_dof_indices_pyb):
                    self._pyb_client.resetJointStateMultiDof(self._pyb_id, m_pyb_ind, _R.from_rotvec(config[m_ind]).as_quat().tolist())
        else:
            raise TypeError("Configuration must be a list or numpy array")


    @property
    def velocity(self):
        velocity = np.zeros((self.dof,))
        if self.__single_dof_indices:
            for s_ind, s_pyb_ind in zip(self.__single_dof_indices, self.__single_dof_indices_pyb):
                velocity[s_ind] = self._pyb_client.getJointState(self._pyb_id, s_pyb_ind)[1]
        if self.__multi_dof_indices:
            m_inds = np.array(self.__multi_dof_indices).reshape(-1, 3)
            for m_ind, m_pyb_ind in zip(m_inds, self.__multi_dof_indices_pyb):
                velocity[m_ind] = np.array(self._pyb_client.getJointStateMultiDof(self._pyb_id, m_pyb_ind)[1])
        return _VelocityArray(self, velocity)


    @velocity.setter
    def velocity(self, velocity: np.ndarray):
        if isinstance(velocity, (list, np.ndarray)):
            if len(velocity) != self.dof:
                raise ValueError(
                    f"Velocity length {len(velocity)} doesn't match number of joints {self.dof}"
                )
            
            if self.__single_dof_indices:
                for s_ind, s_pyb_ind in zip(self.__single_dof_indices, self.__single_dof_indices_pyb):
                    self._pyb_client.resetJointState(
                        self._pyb_id,
                        s_pyb_ind,
                        targetValue=self._pyb_client.getJointState(self._pyb_id, s_pyb_ind)[0],
                        targetVelocity=float(velocity[s_ind]),
                    )
            if self.__multi_dof_indices:
                m_inds = np.array(self.__multi_dof_indices).reshape(-1, 3)
                for m_ind, m_pyb_ind in zip(m_inds, self.__multi_dof_indices_pyb):
                    self._pyb_client.resetJointStateMultiDof(
                        self._pyb_id,
                        m_pyb_ind,
                        targetValue=self._pyb_client.getJointStateMultiDof(self._pyb_id, m_pyb_ind)[0],
                        targetVelocity=velocity[m_ind].tolist(),
                    )
        else:
            raise TypeError("Velocity must be a list or numpy array")


    @property
    def effort(self):
        if not hasattr(self, '_effort'):
            self._effort = np.zeros((self.dof,))
        return _EffortArray(self, self._effort)


    @effort.setter
    def effort(self, effort: np.ndarray):
        if isinstance(effort, (list, np.ndarray)):
            if len(effort) != self.dof:
                raise ValueError(
                    f"Effort length {len(effort)} doesn't match number of joints {self.dof}"
                )
            effort = np.clip(effort, -self.joint_effort_limits, self.joint_effort_limits)
            self._effort = np.copy(effort)
            if self.__single_dof_indices:
                self._pyb_client.setJointMotorControlArray(
                    self._pyb_id,
                    self.__single_dof_indices_pyb,
                    self._pyb_client.TORQUE_CONTROL, 
                    forces=effort[self.__single_dof_indices].tolist(),
                )
            if self.__multi_dof_indices:
                self._pyb_client.setJointMotorControlMultiDofArray(
                    self._pyb_id, 
                    self.__multi_dof_indices_pyb,
                    self._pyb_client.TORQUE_CONTROL, 
                    forces=effort[self.__multi_dof_indices].reshape(-1, 3).tolist(),
                )
        else:
            raise TypeError("Effort must be a list or numpy array")
        

    @property
    def mass_matrix(self):
        if self._static_base:
            return np.array(self._pyb_client.calculateMassMatrix(self._pyb_id, self.configuration.tolist()))
        else:
            return np.array(self._pyb_client.calculateMassMatrix(self._pyb_id, self.base_position.tolist() + _R.from_matrix(self.base_orientation).as_quat().tolist() + self.configuration.tolist()))
    

    @property
    def coriolis_centrifugal_vector(self):
        if self._static_base:
            return np.array(self._pyb_client.calculateInverseDynamics(self._pyb_id, self.configuration.tolist(), self.velocity.tolist(), np.zeros((self.dof,)).tolist())) - self.gravity_vector
        else:
            return np.array(
                self._pyb_client.calculateInverseDynamics(
                    self._pyb_id, 
                    self.base_position.tolist() + _R.from_matrix(self.base_orientation).as_quat().tolist() + self.configuration.tolist(), 
                    self.base_velocity.tolist() + [0.0] + self.velocity.tolist(),
                    np.zeros((7 + self.dof,)).tolist(),
                    flags=1,
                )
            )[self.__floating_base_indices] - self.gravity_vector
        

    @property
    def coriolis_centrifugal_matrix(self):
        def eval_coriolis(qdot):
            if self._static_base:
                return (np.array(self._pyb_client.calculateInverseDynamics(
                    self._pyb_id,
                    self.configuration.tolist(),
                    qdot.tolist(),
                    [0.0] * self.dof,
                )) - self.gravity_vector)
            else:
                return (np.array(self._pyb_client.calculateInverseDynamics(
                    self._pyb_id,
                    self.base_position.tolist() + _R.from_matrix(self.base_orientation).as_quat().tolist() + self.configuration.tolist(),
                    self.base_velocity.tolist() + [0.0] + qdot.tolist(),
                    np.zeros((7 + self.dof,)).tolist(),
                    flags=1,
                ))[self.__floating_base_indices] - self.gravity_vector)
            
        def eval_coriolis_nonvectorized(qdot):
            return np.apply_along_axis(eval_coriolis, axis=0, arr=qdot)
        
        res = jacobian(
            eval_coriolis_nonvectorized, 
            self.velocity,
            step_factor=4,
            maxiter=10,
            order=2,
        )
        if np.any(res.status != 0):
            logger.warning("Jacobian failed to converge in coriolis matrix calculation")
        # We divide by 2 because Coriolis term is quadratic w.r.t. qdot
        return res.df / 2.0 
    

    @property
    def gravity_vector(self):
        if self._static_base:
            return np.array(self._pyb_client.calculateInverseDynamics(self._pyb_id, self.configuration.tolist(), np.zeros((self.dof,)).tolist(), np.zeros((self.dof,)).tolist()))
        else:
            return np.array(
                self._pyb_client.calculateInverseDynamics(
                    self._pyb_id, 
                    self.base_position.tolist() + _R.from_matrix(self.base_orientation).as_quat().tolist() + self.configuration.tolist(), 
                    np.zeros((7 + self.dof,)).tolist(), 
                    np.zeros((7 + self.dof,)).tolist(),
                    flags=1,
                )
            )[self.__floating_base_indices]
        

    @property
    def noninertial_dynamics(self):
        if self._static_base:
            return np.array(self._pyb_client.calculateInverseDynamics(self._pyb_id, self.configuration.tolist(), self.velocity.tolist(), np.zeros((self.dof,)).tolist()))
        else:
            return np.array(
                self._pyb_client.calculateInverseDynamics(
                    self._pyb_id, 
                    self.base_position.tolist() + _R.from_matrix(self.base_orientation).as_quat().tolist() + self.configuration.tolist(), 
                    self.base_velocity.tolist() + [0.0] + self.velocity.tolist(),
                    np.zeros((7 + self.dof,)).tolist(),
                    flags=1,
                )
            )[self.__floating_base_indices]


    def inverse_dynamics(self, qdd: np.ndarray):
        if self._static_base:
            return np.array(
                self._pyb_client.calculateInverseDynamics(
                    self._pyb_id, 
                    self.configuration.tolist(), 
                    self.velocity.tolist(),
                    qdd.tolist()
                )
            ) - self.joint_damping * self.velocity
        else:
            if qdd.shape == (self.dof,):
                qdd = np.hstack((np.zeros((6,)), qdd))
            # Pybullet will set input acceleration to 0 for inverse dynamics when flags=1
            # so we have to solve it manually.
            N = self.noninertial_dynamics
            M = self.mass_matrix
            return (M @ qdd) + N - (np.hstack((np.zeros((6,)), self.joint_damping * self.velocity)))
        

    def inverse_kinematics(
            self,
            position: np.ndarray, 
            orientation: np.ndarray | None = None, 
            seed: np.ndarray | None = None,
            link : int | None = None,
        ):
        return np.array(
            self._pyb_client.calculateInverseKinematics(
                self._pyb_id,
                self._links[link]._index if link is not None else self._links[-1]._index,
                position.tolist(),
                _R.from_matrix(orientation).as_quat().tolist() if orientation is not None else None,
                lowerLimits=self.joint_limits[:, 0].tolist(),
                upperLimits=self.joint_limits[:, 1].tolist(),
                currentPositions=seed.tolist() if seed is not None else self.configuration.tolist(),
                maxNumIterations=100,
                residualThreshold=1e-6,
            )
        )


    def jacobian(self, link: int | None = None, local_position: np.ndarray | None = None):
        Jv, Jw = self._pyb_client.calculateJacobian(
            self._pyb_id,
            self._links[link]._index if link is not None else self._links[-1]._index,
            local_position.tolist() if local_position is not None else [0, 0, 0],
            self.configuration.tolist(),
            [0] * self.dof,
            [0] * self.dof,
        )
        return np.vstack((np.array(Jv), np.array(Jw)))


    def forward_kinematics(self, link: int | None = None, local_position: np.ndarray = np.eye(4)):
        return self._links[link if link is not None else -1].pose @ local_position
    

    @property
    def contacts(self):
        contact_list = []
        contacts = self._pyb_client.getContactPoints(self._pyb_id)
        for contact in contacts:
            subject = self._links_from_indices[contact[3]]
            if (target := self.world._pyb_entity_map.get(contact[2])) is None:
                raise ValueError(f"Target {contact[2]} is not a valid object in the world")
            if isinstance(target, Chain):
                target = target._links_from_indices[contact[4]]
            
            contact_list.append(
                physics.Contact(
                    subject=subject,
                    target=target,
                    position_on_subject=np.array(contact[5]),
                    position_on_target=np.array(contact[6]),
                    contact_normal_target_to_subject=np.array(contact[7]),
                    contact_distance=contact[8],
                    normal_force=contact[9],
                    lateral_friction_a=contact[10],
                    lateral_friction_a_dir=np.array(contact[11]),
                    lateral_friction_b=contact[12],
                    lateral_friction_b_dir=np.array(contact[13]),
                )
            )
        return contact_list
    

    def distance_to(self, target: 'physics.PhysicsObject | Link | Chain', max_distance: float = np.inf):
        proximity_list = []
        distances = self._pyb_client.getClosestPoints(
            self._pyb_id, 
            target._pyb_id, 
            max_distance, 
            **({'linkIndexB': target._index} if isinstance(target, Link) else {}),
        )
        for distance in distances:
            subject = self._links_from_indices[distance[3]]
            if (target := self.world._pyb_entity_map.get(distance[2])) is None:
                raise ValueError(f"Target {distance[2]} is not a valid object in the world")
            if isinstance(target, Chain):
                target = target._links_from_indices[distance[4]]
            
            proximity_list.append(
                physics.Proximity(
                    subject=subject,
                    target=target,
                    position_on_subject=np.array(distance[5]),
                    position_on_target=np.array(distance[6]),
                    normal_target_to_subject=np.array(distance[7]),
                    distance=distance[8],
                )
            )
        return proximity_list
    

    @property
    def base_pose(self):
        p, q = self._pyb_client.getBasePositionAndOrientation(self._pyb_id)
        return trans_mat(pos=np.array(p), orn=_R.from_quat(q).as_matrix())


    @base_pose.setter
    def base_pose(self, T: np.ndarray):
        self._pyb_client.resetBasePositionAndOrientation(self._pyb_id, T[:3, 3].tolist(), _R.from_matrix(T[:3, :3]).as_quat().tolist())


    @property
    def base_position(self):
        p, _ = self._pyb_client.getBasePositionAndOrientation(self._pyb_id)
        return np.array(p)


    @base_position.setter
    def base_position(self, p: np.ndarray):
        self._pyb_client.resetBasePositionAndOrientation(self._pyb_id, p.tolist(), _R.from_matrix(self.base_orientation).as_quat().tolist())


    @property
    def base_orientation(self):
        _, q = self._pyb_client.getBasePositionAndOrientation(self._pyb_id)
        return _R.from_quat(q).as_matrix()


    @base_orientation.setter
    def base_orientation(self, q: np.ndarray):
        self._pyb_client.resetBasePositionAndOrientation(self._pyb_id, self.base_position.tolist(), _R.from_matrix(q).as_quat().tolist())


    @property
    def base_velocity(self):
        v, w = self._pyb_client.getBaseVelocity(self._pyb_id)
        return np.hstack((np.array(v), np.array(w)))


    @base_velocity.setter
    def base_velocity(self, v: np.ndarray):
        self._pyb_client.resetBaseVelocity(self._pyb_id, v[:3].tolist(), v[3:].tolist())


    @property
    def base_linear_velocity(self):
        v, _ = self._pyb_client.getBaseVelocity(self._pyb_id)
        return np.array(v)


    @base_linear_velocity.setter
    def base_linear_velocity(self, v: np.ndarray):
        self._pyb_client.resetBaseVelocity(self._pyb_id, linearVelocity=v.tolist())


    @property
    def base_angular_velocity(self):
        _, w = self._pyb_client.getBaseVelocity(self._pyb_id)
        return np.array(w)


    @base_angular_velocity.setter
    def base_angular_velocity(self, w: np.ndarray):
        self._pyb_client.resetBaseVelocity(self._pyb_id, angularVelocity=w.tolist())



class SerialChain(Chain):
    def __init__(
        self,
        links: list[Link],
        joints: list[Joint],
        static_base: bool = False,
        **kwargs,
    ):
        world = physics.get_world(name=kwargs.pop('world_name', None))
        self._world_name = world.name
        self._pyb_client = SimpleBulletClient(world._pyb_client._client)

        self._pyb_id = None
        _links = copy.deepcopy(links)
        _joints = copy.deepcopy(joints)

        if len(_links) != len(_joints) + 1:
            raise ValueError("Number of links must be one more than number of joints")
        
        base_link = _links[0]

        base_id = _create_multibody(
            world=world,
            static_base=static_base,
            base_link=base_link,
            links=_links[1:],
            joints=_joints,
            link_masses=[link._mass for link in _links[1:]],
            link_positions=[joint.offset[:3, 3].tolist() for joint in _joints],
            link_orientations=[_R.from_matrix(joint.offset[:3, :3]).as_quat().tolist() for joint in _joints],
            link_parent_indices=[i for i in range(len(_links) - 1)],
            link_inertial_positions=[link._offset[:3, 3].tolist() for link in _links[1:]],
            link_inertial_orientations=[_R.from_matrix(link._offset[:3, :3]).as_quat().tolist() for link in _links[1:]],
            joint_types=[joint.type.value for joint in _joints],
            joint_axes=[joint.axis.tolist() for joint in _joints],
        )

        for parent, child in zip(range(len(links) - 1), range(1, len(links))):
            self._pyb_client.setCollisionFilterPair(base_id, base_id, parent, child, 0)

        super().__init__(base_id, _links, _joints, static_base, **kwargs)



class GraphChain(Chain):
    def __init__(self, G: nx.DiGraph, static_base: bool = False, **kwargs):

        world = physics.get_world(name=kwargs.pop('world_name', None))
        self._world_name = world.name
        self._pyb_client = SimpleBulletClient(world._pyb_client._client)

        self._pyb_id = None
        if not nx.is_tree(G):
            raise ValueError("Chain graph must be a tree")

        if not nx.is_directed_acyclic_graph(G):
            raise ValueError("Chain graph must be a directed acyclic graph")

        G = copy.deepcopy(G)

        G_sorted = nx.topological_sort(G)
        root_node = next(iter(G_sorted))

        base_link = G.nodes[root_node]["link"]
        if root_node != base_link._name:
            raise ValueError(f"Base link name {base_link._name} must match root node name {root_node}")
        
        node_to_index = {root_node: 0}
        
        links = []
        joints = []
        link_masses = []
        link_positions = []
        link_orientations = []
        link_parent_indices = []
        link_inertial_positions = []
        link_inertial_orientations = []
        joint_types = []
        joint_axes = []

        for node in G_sorted:
            if node == root_node:
                continue
                
            parents = list(G.predecessors(node))
            if len(parents) != 1:
                raise ValueError(f"Node {node} has {len(parents)} parents, expected 1")
            parent = parents[0]
            
            link = G.nodes[node]["link"]
            joint = G.edges[(parent, node)]["joint"]

            if link._name != node:
                raise ValueError(f"Link name {link._name} must match node name {node}")
            
            links.append(link)
            joints.append(joint)
            link_masses.append(link._mass)
            link_positions.append(joint.offset[:3, 3].tolist())
            link_orientations.append(_R.from_matrix(joint.offset[:3, :3]).as_quat().tolist())
            link_parent_indices.append(node_to_index[parent])
            link_inertial_positions.append(link._offset[:3, 3].tolist())
            link_inertial_orientations.append(_R.from_matrix(link._offset[:3, :3]).as_quat().tolist())
            joint_types.append(joint.type.value)
            joint_axes.append(joint.axis.tolist())
            
            node_to_index[node] = len(links)
        
        links = [base_link] + links
        joints = joints

        base_id = _create_multibody(
            world=world,
            static_base=static_base,
            base_link=links[0],
            links=links[1:],
            joints=joints,
            link_masses=link_masses,
            link_positions=link_positions,
            link_orientations=link_orientations,
            link_parent_indices=link_parent_indices,
            link_inertial_positions=link_inertial_positions,
            link_inertial_orientations=link_inertial_orientations,
            joint_types=joint_types,
            joint_axes=joint_axes,
        )

        G.nodes[root_node]["link"] = base_link
        for link, node in zip(links[1:], G_sorted):
            if node == root_node:
                continue
                
            parent = list(G.predecessors(node))[0]
            G.nodes[node]["link"] = link

            parent_idx = node_to_index[parent]
            node_idx = node_to_index[node]
            
            if parent_idx >= 0:
                self._pyb_client.setCollisionFilterPair(base_id, base_id, parent_idx, node_idx, 0)

        self._G = G
        
        super().__init__(base_id, links, joints, static_base, **kwargs)


    @classmethod
    def from_urdf(cls, urdf_path: str):
        return _chain_from_urdf(urdf_path)


    @classmethod
    def inferred_attrs(cls):
        return super().inferred_attrs() + ['links', 'joints']