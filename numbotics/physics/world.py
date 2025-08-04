import weakref
import time
from typing import Callable, Any
import contextlib

import numpy as np
from scipy.spatial.transform import Rotation as _R
from pybullet_utils.bullet_client import BulletClient

from numbotics.physics import pyb
from numbotics.graphics import Visualizer
from numbotics.math import trans_mat
from numbotics.utils import Shape, pipes
from .object import PhysicsObject, Cube
from .chain import Chain, Link
from .constraint import Constraint, Joint


WORLD_INSTANCES: dict[str, 'World'] = {}
SELECTED_WORLD: str | None = None



def get_world(name: str | None = None):
    global WORLD_INSTANCES, SELECTED_WORLD
    if name is None:
        name = SELECTED_WORLD
    if name is None and len(WORLD_INSTANCES) == 0:
        name = 'world_0'
    if name is None:
        raise ValueError('There should always be a selected World instance if a World exists')
    if name not in WORLD_INSTANCES:
        WORLD_INSTANCES[name] = World(name=name)
    SELECTED_WORLD = name
    return WORLD_INSTANCES[name]



class World:

    def __init__(self, name: str | None = None, visualize: bool = False):
        global WORLD_INSTANCES, SELECTED_WORLD
        # If we use normal dicts, objects created in functions and other
        # local scopes will not be deleted automatically because the
        # references will be tracked by these sets.
        if name is None:
            self._name = f'world_{len(WORLD_INSTANCES)}'
        else:
            self._name = name
        if name in WORLD_INSTANCES:
            raise ValueError(f'World with name {name} already exists')
        WORLD_INSTANCES[self._name] = self
        SELECTED_WORLD = self._name

        self._static_objects = weakref.WeakValueDictionary()
        self._dynamic_objects = weakref.WeakValueDictionary()
        self._pyb_entity_map = weakref.WeakValueDictionary()
        self._callbacks = weakref.WeakValueDictionary()
        self._constraints = {}

        self._vis = Visualizer() if visualize else None

        with pipes():
            self._pyb_client = BulletClient(connection_mode=pyb.DIRECT)


    def __del__(self):
        for body in list(self._static_objects.values()) + list(self._dynamic_objects.values()):
            self.unregister(body)
        
        self._static_objects = None
        self._dynamic_objects = None
        self._pyb_entity_map = None
        self._callbacks = None
        self._constraints = None

        if self._vis is not None:
            self._vis.close()
            self._vis = None
        
        self._pyb_client.disconnect()
        self._pyb_client = None


    @property
    def name(self):
        return self._name


    @property
    def visualizer_url(self):
        if self._vis is not None:
            return self._vis.url()
        else:
            return None


    @contextlib.contextmanager
    def pool(self, poolsize: int = 1):
        global WORLD_INSTANCES
        
        worlds = weakref.WeakSet()
        world_refs = []
        bodies = []
        p_idx = 0
        
        for _ in range(poolsize):
            while (world_name := f'{self.name}_subworld_{p_idx}') in WORLD_INSTANCES:
                p_idx += 1
            
            world = World(name=world_name)            
            worlds.add(world)
            world_refs.append(world)

            for _, body in sorted(list((key, value) for key, value in self._pyb_entity_map.items()), key=lambda x: x[0]):

                _cls = type(body)
            
                body_attrs = body.__dict__.copy()
                body_attrs = {key.strip('_'): value for key, value in body_attrs.items()}
                for attr in _cls.inferred_attrs():
                    body_attrs.pop(attr, None)
                body_attrs['world_name'] = world_refs[-1].name                

                if isinstance(body, PhysicsObject):
                    geom_info = body._collision_shape._shape_info.copy()
                    if isinstance(body, Cube):
                        geom_info['half_extent'] = geom_info['half_extents'][0]
                        del geom_info['half_extents']

                    bodies.append(_cls(**body_attrs, **geom_info))

                    bodies[-1].pose = body.pose
                    bodies[-1].velocity = body.velocity

                elif isinstance(body, Chain):
                    bodies.append(_cls(**body_attrs))

                    bodies[-1].base_pose = body.base_pose
                    bodies[-1].base_velocity = body.base_velocity
                    
                    bodies[-1].configuration = body.configuration
                    bodies[-1].velocity = body.velocity
                    bodies[-1].effort = body.effort

            p_idx += 1

        try:
            yield worlds
        
        finally:
            del bodies
            for world in world_refs:
                del WORLD_INSTANCES[world.name]
            del worlds
            del world_refs


    def step(self, sleep: bool = False):
        step_start = time.time()

        for callback in list(self._callbacks.values()):
            callback()

        self._pyb_client.stepSimulation()
        self.update_visualizer()
        
        step_end = time.time()
        if sleep and (step_end - step_start) < self.dt:
            time.sleep(self.dt - (step_end - step_start))


    def step_collision_detection(self):
        self._pyb_client.performCollisionDetection()
        self.update_visualizer()
                

    def update_visualizer(self):
        
        if self._vis is not None:

            for body in list(self._static_objects.values()) + list(self._dynamic_objects.values()):
                if isinstance(body, PhysicsObject):
                    T = body.pose
                    if body._visual_shape.shape == Shape.CYLINDER:
                        T = T @ trans_mat(orn=_R.from_euler("x", -np.pi / 2).as_matrix())
                    self._vis.set_transform(body.name, T)
                elif isinstance(body, Chain):
                    poses = body.poses
                    for link, T in zip(body._links, poses):
                        if link._visual_shape.shape == Shape.EMPTY:
                            continue
                        T = T @ link._visual_shape.offset
                        if link._visual_shape.shape == Shape.CYLINDER:
                            # Correct different default orientation for cylinders
                            # between PyBullet and Meshcat
                            T = T @ trans_mat(orn=_R.from_euler("x", -np.pi / 2).as_matrix())
                        self._vis.set_transform(link.name, T)

                else:
                    raise ValueError(f"Unknown body type: {type(body)}")


    def clear(self):
        self._static_objects = weakref.WeakValueDictionary()
        self._dynamic_objects = weakref.WeakValueDictionary()
        self._pyb_entity_map = weakref.WeakValueDictionary()
        self._callbacks = weakref.WeakValueDictionary()
        self._constraints = {}
        self._pyb_client.resetSimulation()


    def get_object(self, name: str, default: Any = None):
        if name in self._static_objects:
            return self._static_objects[name]
        elif name in self._dynamic_objects:
            return self._dynamic_objects[name]
        else:
            # fallback to search for links. NOTE: chains cannot be static.
            for obj in list(self._dynamic_objects.values()):
                if isinstance(obj, Chain):
                    for link in obj._links:
                        if link.name == name:
                            return link
        return default


    @property
    def gravity(self):
        pass


    @gravity.setter
    def gravity(self, g: np.ndarray):
        self._pyb_client.setGravity(g[0], g[1], g[2])


    @property
    def dt(self):
        if not hasattr(self, '_dt'):
            self._dt = self._pyb_client.getPhysicsEngineParameters()["fixedTimeStep"]
        return self._dt


    @dt.setter
    def dt(self, dt: float):
        self._dt = dt
        self._pyb_client.setPhysicsEngineParameter(fixedTimeStep=dt)


    def register(self, body):
        if not isinstance(body, PhysicsObject) and not isinstance(body, Chain):
            raise ValueError(f"Unknown body type: {type(body)}")

        if body._static:
            self._static_objects[body.name] = body
        else:
            self._dynamic_objects[body.name] = body
        self._pyb_entity_map[body._pyb_id] = body

        if self._vis is not None:
            
            if isinstance(body, PhysicsObject):
                T = body.pose
                if body._visual_shape.shape == Shape.CYLINDER:
                    T = T @ trans_mat(orn=_R.from_euler("x", -np.pi / 2).as_matrix())
                self._vis.register(body.name, body._visual_shape.visual_shape)
                self._vis.set_transform(body.name, T)
                self._vis.set_color(body.name, body._visual_shape.color)
            
            elif isinstance(body, Chain):
                poses = body.poses
                for link, T in zip(body._links, poses):
                    if link._visual_shape.shape == Shape.EMPTY:
                        continue
                    
                    T = T @ link._visual_shape.offset
                    if link._visual_shape.shape == Shape.CYLINDER:
                        # Correct different default orientation for cylinders
                        # between PyBullet and Meshcat
                        T = T @ trans_mat(orn=_R.from_euler("x", -np.pi / 2).as_matrix())

                    self._vis.register(link.name, link._visual_shape.visual_shape)
                    self._vis.set_transform(link.name, T)
                    self._vis.set_color(link.name, link._visual_shape.color)


    def unregister(self, body):
        if not isinstance(body, PhysicsObject) and not isinstance(body, Chain):
            raise ValueError(f"Unknown body type: {type(body)}")
        
        if self._vis is not None:
            if isinstance(body, PhysicsObject):
                self._vis.unregister(body.name)
            elif isinstance(body, Chain):
                for link in body._links:
                    self._vis.unregister(link.name)

        if body in self._static_objects:
            del self._static_objects[body.name]
        elif body in self._dynamic_objects:
            del self._dynamic_objects[body.name]
        del self._pyb_entity_map[body._pyb_id]

        if f'callback_{body.name}' in self._callbacks:
            del self._callbacks[f'callback_{body.name}']


    def add_callback(self, callback: Callable):
        if not isinstance(callback, Callable):
            raise ValueError(f"Callback must be a callable, got {type(callback)}")
        self._callbacks[f'callback_{callback._body.name}'] = callback


    def add_constraint(self, body1, body2, constraint: Joint):
        if constraint.type not in (Constraint.FIXED, Constraint.PRISMATIC):
            raise NotImplementedError(
                f"Only fixed and prismatic constraints can be added programmatically, got {constraint.type}"
            )

        if isinstance(body1, Chain):
            body1_id = body1._pyb_id
            body1_link_index = -1
        elif isinstance(body1, Link):
            body1_id = body1._body_id
            body1_link_index = body1._index if body1._index is not None else -1
        elif isinstance(body1, PhysicsObject):
            body1_id = body1._pyb_id
            body1_link_index = -1

        if isinstance(body2, Chain):
            body2_id = body2._pyb_id
            body2_link_index = -1
        elif isinstance(body2, Link):
            body2_id = body2._body_id
            body2_link_index = body2._index if body2._index is not None else -1
        elif isinstance(body2, PhysicsObject):
            body2_id = body2._pyb_id
            body2_link_index = -1

        parent_pose = constraint._offset @ constraint._parent_pose
        child_pose = constraint._offset @ constraint._child_pose
        self._constraints[(body1_id, body1_link_index, body2_id, body2_link_index)] = (
            self._pyb_client.createConstraint(
                parentBodyUniqueId=body1_id,
                parentLinkIndex=body1_link_index,
                childBodyUniqueId=body2_id,
                childLinkIndex=body2_link_index,
                jointType=constraint.type.value,
                jointAxis=constraint.axis.tolist(),
                parentFramePosition=parent_pose[:3, 3].tolist(),
                childFramePosition=child_pose[:3, 3].tolist(),
                parentFrameOrientation=_R.from_matrix(parent_pose[:3, :3])
                .as_quat()
                .tolist(),
                childFrameOrientation=_R.from_matrix(child_pose[:3, :3])
                .as_quat()
                .tolist(),
            )
        )


    def depth_image(
            self, 
            width: int, 
            height: int, 
            camera_pose: np.ndarray,
            near: float = 0.01,
            far: float = 1000,
            fov: float = 60,
        ):
        # Camera is based at camera pose position, looking through x-axis,
        # with z-axis representing the "up" direction.
        view_mat = self._pyb_client.computeViewMatrix(
            cameraEyePosition=camera_pose[:3, 3].tolist(),
            cameraTargetPosition=camera_pose[:3, 0].tolist(),
            cameraUpVector=camera_pose[:3, 2].tolist(),
        )

        proj_mat = self._pyb_client.computeProjectionMatrixFOV(
            fov=fov,
            aspect=width / height,
            nearVal=near,
            farVal=far,
        )

        image = self._pyb_client.getCameraImage(
            width, 
            height, 
            viewMatrix=view_mat, 
            projectionMatrix=proj_mat, 
            renderer=self._pyb_client.ER_BULLET_HARDWARE_OPENGL,
            flags=self._pyb_client.ER_NO_SEGMENTATION_MASK,
        )

        depth_img = np.array(image[3]).reshape(height, width, 1)

        return (far * near) / (far - (far - near) * depth_img)
