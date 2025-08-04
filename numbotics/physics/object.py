import enum

import numpy as np
from scipy.spatial.transform import Rotation as _R

import numbotics.utils.logger as logger
from numbotics.physics import pyb
import numbotics.physics as physics
from numbotics.graphics import VisualShape
from numbotics.math import trans_mat
from numbotics.utils import Shape, parse_shape_kwargs
from numbotics.physics.helpers import SimpleBulletClient


class PhysicsObject:

    def __init__(
            self, 
            mass: float,
            static: bool,
            collision_shape: 'physics.CollisionShape | None' = None,
            visual_shape: 'VisualShape | None' = None,
            **kwargs,
        ):
        if mass < 0:
            raise ValueError("Link mass must be positive")
        
        self._mass = mass if not static else 0
        self._static = static

        world = physics.get_world(name=kwargs.pop('world_name', None))

        self._collision_shape = collision_shape if collision_shape is not None else physics.CollisionShape(Shape.EMPTY)
        self._visual_shape = visual_shape if visual_shape is not None else VisualShape(Shape.EMPTY)
        if world._vis is not None:
            if self._visual_shape.shape == Shape.EMPTY and self._collision_shape.shape != Shape.EMPTY:
                visual_shape_type = self._collision_shape.shape
                if visual_shape_type is None:
                    raise ValueError(f"Invalid shape type: {self._collision_shape.shape}")
                self._visual_shape = VisualShape(
                    visual_shape_type,
                    **self._collision_shape._shape_info,
                )

        self._world_name = world.name
        self._pyb_client = SimpleBulletClient(world._pyb_client._client)

        self._collision_shape.register(pyb_client=self._pyb_client)
        self._pyb_id = self._pyb_client.createMultiBody(
            baseMass=self._mass, 
            baseCollisionShapeIndex=self._collision_shape.col_id,
        )

        self._name = kwargs.pop('name', f'physics_object_{self._pyb_id}')

        for key in list(kwargs.keys()):
            if hasattr(self, f"{key}"):
                setattr(self, f"{key}", kwargs.pop(key))
        
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

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
        return f"entity_id_{self._pyb_id}"


    def __eq__(self, other):
        return str(self) == str(other)


    def __hash__(self):
        return hash(self.name)
    

    @classmethod
    def inferred_attrs(cls):
        return ['pyb_id', 'pyb_client']
    

    @property
    def name(self):
        return f'{self._world_name}:{self._name}'


    @property
    def mass(self):
        return self._pyb_client.getDynamicsInfo(self._pyb_id, -1)[0]
    

    @mass.setter
    def mass(self, mass: float):
        self._mass = mass
        self._pyb_client.changeDynamics(self._pyb_id, -1, mass=mass)


    @property
    def linear_damping(self):
        if not hasattr(self, '_linear_damping'):
            self._linear_damping = 0.04
        return self._linear_damping


    @linear_damping.setter
    def linear_damping(self, linear_damping: float):
        self._linear_damping = linear_damping
        self._pyb_client.changeDynamics(self._pyb_id, -1, linearDamping=linear_damping)


    @property
    def angular_damping(self):
        if not hasattr(self, '_angular_damping'):
            self._angular_damping = 0.04
        return self._angular_damping


    @angular_damping.setter
    def angular_damping(self, angular_damping: float):
        self._angular_damping = angular_damping
        self._pyb_client.changeDynamics(self._pyb_id, -1, angularDamping=angular_damping)


    @property
    def restitution(self):
        return self._pyb_client.getDynamicsInfo(self._pyb_id, -1)[5]
    

    @restitution.setter
    def restitution(self, restitution: float):
        self._pyb_client.changeDynamics(self._pyb_id, -1, restitution=restitution)


    @property
    def lateral_friction(self):
        return self._pyb_client.getDynamicsInfo(self._pyb_id, -1)[1]


    @lateral_friction.setter
    def lateral_friction(self, lateral_friction: float):
        self._pyb_client.changeDynamics(self._pyb_id, -1, lateralFriction=lateral_friction)


    @property
    def rolling_friction(self):
        return self._pyb_client.getDynamicsInfo(self._pyb_id, -1)[6]


    @rolling_friction.setter
    def rolling_friction(self, rolling_friction: float):
        self._pyb_client.changeDynamics(self._pyb_id, -1, rollingFriction=rolling_friction)


    @property
    def spinning_friction(self):
        return self._pyb_client.getDynamicsInfo(self._pyb_id, -1)[7]


    @spinning_friction.setter
    def spinning_friction(self, spinning_friction: float):
        self._pyb_client.changeDynamics(self._pyb_id, -1, spinningFriction=spinning_friction)


    @property
    def contact_damping(self):
        return self._pyb_client.getDynamicsInfo(self._pyb_id, -1)[8]


    @contact_damping.setter
    def contact_damping(self, contact_damping: float):
        self._pyb_client.changeDynamics(self._pyb_id, -1, contactDamping=contact_damping)


    @property
    def contact_stiffness(self):
        return self._pyb_client.getDynamicsInfo(self._pyb_id, -1)[9]


    @contact_stiffness.setter
    def contact_stiffness(self, contact_stiffness: float):
        self._pyb_client.changeDynamics(self._pyb_id, -1, contactStiffness=contact_stiffness)


    @property
    def pose(self):
        p, q = self._pyb_client.getBasePositionAndOrientation(self._pyb_id)
        return trans_mat(pos=np.array(p), orn=_R.from_quat(q).as_matrix())


    @pose.setter
    def pose(self, T: np.ndarray):
        self._pyb_client.resetBasePositionAndOrientation(
            self._pyb_id,
            T[:3, 3].tolist(),
            _R.from_matrix(T[:3, :3]).as_quat().tolist(),
        )


    @property
    def position(self):
        p, _ = self._pyb_client.getBasePositionAndOrientation(self._pyb_id)
        return np.array(p)


    @position.setter
    def position(self, p: np.ndarray):
        self._pyb_client.resetBasePositionAndOrientation(
            self._pyb_id,
            p.tolist(),
            _R.from_matrix(self.orientation).as_quat().tolist(),
        )


    @property
    def orientation(self):
        _, q = self._pyb_client.getBasePositionAndOrientation(self._pyb_id)
        return _R.from_quat(q).as_matrix()


    @orientation.setter
    def orientation(self, R: np.ndarray):
        self._pyb_client.resetBasePositionAndOrientation(
            self._pyb_id, 
            self.position.tolist(), 
            _R.from_matrix(R).as_quat().tolist()
        )


    @property
    def velocity(self):
        v, w = self._pyb_client.getBaseVelocity(self._pyb_id)
        return np.hstack((np.array(v), np.array(w)))


    @velocity.setter
    def velocity(self, v: np.ndarray):
        self._pyb_client.resetBaseVelocity(self._pyb_id, v[:3].tolist(), v[3:].tolist())


    @property
    def linear_velocity(self):
        v, _ = self._pyb_client.getBaseVelocity(self._pyb_id)
        return np.array(v)


    @linear_velocity.setter
    def linear_velocity(self, v: np.ndarray):
        self._pyb_client.resetBaseVelocity(self._pyb_id, linearVelocity=v.tolist())


    @property
    def angular_velocity(self):
        _, w = self._pyb_client.getBaseVelocity(self._pyb_id)
        return np.array(w)


    @angular_velocity.setter
    def angular_velocity(self, w: np.ndarray):
        self._pyb_client.resetBaseVelocity(self._pyb_id, angularVelocity=w.tolist())


    def apply_wrench(self, wrench: np.ndarray, position: np.ndarray = np.zeros((3,)), local: bool = False):
        self._pyb_client.applyExternalForce(
            self._pyb_id, -1, wrench[:3].tolist(), position.tolist(), self._pyb_client.LINK_FRAME if local else self._pyb_client.WORLD_FRAME
        )
        self._pyb_client.applyExternalTorque(
            self._pyb_id, -1, wrench[3:].tolist(), self._pyb_client.LINK_FRAME if local else self._pyb_client.WORLD_FRAME
        )


    def apply_force(self, force: np.ndarray, position: np.ndarray = np.zeros((3,)), local: bool = False):
        self._pyb_client.applyExternalForce(
            self._pyb_id, -1, force.tolist(), position.tolist(), self._pyb_client.LINK_FRAME if local else self._pyb_client.WORLD_FRAME
        )


    def apply_torque(self, torque: np.ndarray, local: bool = False):
        self._pyb_client.applyExternalTorque(self._pyb_id, -1, torque.tolist(), self._pyb_client.LINK_FRAME if local else self._pyb_client.WORLD_FRAME)


    @property
    def contacts(self):
        contact_list = []
        contacts = self._pyb_client.getContactPoints(self._pyb_id, -1)
        for contact in contacts:
            if (target := physics.World()._pyb_entity_map.get(contact[2])) is None:
                raise ValueError(f"Target {contact[2]} is not a valid object in the world")
            if isinstance(target, physics.Chain):
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


    def distance_to(self, target: 'PhysicsObject | physics.Link | physics.Chain', max_distance: float = np.inf):
        proximity_list = []
        distances = self._pyb_client.getClosestPoints(
            self._pyb_id, 
            target._pyb_id, 
            max_distance, 
            **({'linkIndexB': target._index} if isinstance(target, physics.Link) else {}),
        )
        for distance in distances:
            if (target := physics.World()._pyb_entity_map.get(distance[2])) is None:
                raise ValueError(f"Target {distance[2]} is not a valid object in the world")
            if isinstance(target, physics.Chain):
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



class Cube(PhysicsObject):

    def __init__(self, mass: float, half_extent: float, static: bool = False, **kwargs):
        if mass > 0.0 and static:
            logger.warning('Cube is static, mass will be ignored...')
        kwargs, shape_info = parse_shape_kwargs(kwargs)
        super().__init__(
            mass, 
            static, 
            physics.CollisionShape(
                Shape.CUBE,
                half_extents=np.array([half_extent, half_extent, half_extent]),
                **shape_info,
            ),
            None,
            **kwargs,
        )


    @classmethod
    def inferred_attrs(cls):
        return super().inferred_attrs() + ['collision_shape', 'visual_shape']



class Cuboid(PhysicsObject):

    def __init__(self, mass: float, half_extents: np.ndarray, static: bool = False, **kwargs):
        if mass > 0.0 and static:
            logger.warning('Cuboid is static, mass will be ignored...')
        kwargs, shape_info = parse_shape_kwargs(kwargs)
        super().__init__(
            mass,
            static,
            physics.CollisionShape(
                Shape.CUBOID,
                half_extents=half_extents,
                **shape_info,
            ),
            None,
            **kwargs,
        )


    @classmethod
    def inferred_attrs(cls):
        return super().inferred_attrs() + ['collision_shape', 'visual_shape']



class Sphere(PhysicsObject):

    def __init__(self, mass: float, radius: float, static: bool = False, **kwargs):
        if mass > 0.0 and static:
            logger.warning("Sphere is static, mass will be ignored...")
        kwargs, shape_info = parse_shape_kwargs(kwargs)
        super().__init__(
            mass,
            static,
            physics.CollisionShape(
                Shape.SPHERE,
                radius=radius,
                **shape_info,
            ),
            None,
            **kwargs,
        )


    @classmethod
    def inferred_attrs(cls):
        return super().inferred_attrs() + ['collision_shape', 'visual_shape']



class Mesh(PhysicsObject):

    def __init__(self, mass: float, filename: str, static: bool = False, **kwargs):
        self._filename = filename
        if mass > 0.0 and static:
            logger.warning("Mesh is static, mass will be ignored...")
        kwargs, shape_info = parse_shape_kwargs(kwargs)
        super().__init__(
            mass,
            static,
            physics.CollisionShape(
                Shape.MESH,
                filename=filename,
                **shape_info,
            ),
            None,
            **kwargs,
        )


    @classmethod
    def inferred_attrs(cls):
        return super().inferred_attrs() + ['collision_shape', 'visual_shape']



class Plane(PhysicsObject):

    def __init__(self, mass: float, normal: np.ndarray, static: bool = False, **kwargs):
        if mass > 0.0 and static:
            logger.warning("Plane is static, mass will be ignored...")
        kwargs, shape_info = parse_shape_kwargs(kwargs)
        super().__init__(
            mass,
            static,
            physics.CollisionShape(
                Shape.PLANE, 
                normal=normal,
                **shape_info,
            ),
            None,
            **kwargs,
        )


    @classmethod
    def inferred_attrs(cls):
        return super().inferred_attrs() + ['collision_shape', 'visual_shape']



class Capsule(PhysicsObject):

    def __init__(self, mass: float, radius: float, height: float, static: bool = False, **kwargs):
        if mass > 0.0 and static:
            logger.warning("Capsule is static, mass will be ignored...")
        kwargs, shape_info = parse_shape_kwargs(kwargs)
        super().__init__(
            mass,
            static,
            physics.CollisionShape(
                Shape.CAPSULE,
                radius=radius,
                height=height,
                **shape_info,
            ),
            None,
            **kwargs,
        )


    @classmethod
    def inferred_attrs(cls):
        return super().inferred_attrs() + ['collision_shape', 'visual_shape']



class Cylinder(PhysicsObject):

    def __init__(self, mass: float, radius: float, height: float, static: bool = False, **kwargs):
        if mass > 0.0 and static:
            logger.warning("Cylinder is static, mass will be ignored...")
        kwargs, shape_info = parse_shape_kwargs(kwargs)
        super().__init__(
            mass,
            static,
            physics.CollisionShape(
                Shape.CYLINDER,
                radius=radius,
                height=height,
                **shape_info,
            ),
            None,
            **kwargs,
        )


    @classmethod
    def inferred_attrs(cls):
        return super().inferred_attrs() + ['collision_shape', 'visual_shape']