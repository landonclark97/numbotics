from abc import ABC, abstractmethod
import functools
import inspect

import numpy as np
from scipy.spatial.transform import Rotation as _R
import networkx as nx

from numbotics.physics import pyb
import numbotics.physics as phys
import numbotics.graphics as gfx
from numbotics.math import trans_mat
from numbotics.utils import Shape



# This class is needed because when a normal bullet client is garbage
# collected, it shuts down the server. As such, only World objects
# hold normal bullet clients, and these SimpleBulletClients are given
# to Chains, Links, and PhysicsObjects.
class SimpleBulletClient:

    def __init__(self, client_id: int):
        self._client = client_id

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pyb, name)
        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self._client)
        if attribute == 'disconnect':
            self._client = -1
        return attribute



class _IndexableArray(np.ndarray):
    
    def __new__(cls, chain, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.chain = chain
        return obj


    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.chain = getattr(obj, "chain", None)

    
    @abstractmethod
    def __set_element(self, key, value):
        raise NotImplementedError


    def __setitem__(self, key, value, set_func=None):
        if set_func is None:
            set_func = self.__set_element
        if isinstance(key, slice):
            key = range(*key.indices(self.chain.dof))
        if isinstance(key, (int, np.integer)):
            if key >= self.chain.dof:
                raise ValueError(
                    f"Key {key} is greater than the number of joints {self.chain.dof}"
                )
            elif key in self.chain._Chain__single_dof_indices:
                if not isinstance(value, (float, int)):
                    raise ValueError(f"Value {value} is not a float or int")
                set_func(key, value)
            elif key in self.chain._Chain__multi_dof_indices:
                raise NotImplementedError("Multi-dof joint access is not supported")
        elif isinstance(key, (range, list, np.ndarray)):
            if len(key) > self.chain.dof:
                raise ValueError(
                    f"Key length {len(key)} is greater than the number of joints {self.chain.dof}"
                )
            if isinstance(value, (list, np.ndarray)):
                if len(value) != len(key):
                    raise ValueError(f"Value {value} is not a list or numpy array of the same length as key {key}")
            for i, idx in enumerate(list(key)):
                if idx >= self.chain.dof:
                    raise ValueError(
                        f"Key {idx} is greater than the number of joints {self.chain.dof}"
                    )
                if idx in self.chain._Chain__single_dof_indices:
                    if isinstance(value, (list, np.ndarray)):
                        val = value[i]
                    else:
                        val = value
                    set_func(idx, val)
                elif idx in self.chain._Chain__multi_dof_indices:
                    raise NotImplementedError("Multi-dof joint access is not supported")
        else:
            raise ValueError(f"Invalid key type: {type(key)}")



class _ConfigurationArray(_IndexableArray):

    def __set_element(self, key, value):
        self.chain._pyb_client.resetJointState(self.chain._pyb_id, self.chain._Chain__nonfixed_indices_pyb[key], float(value))

    def __setitem__(self, key, value):
        super().__setitem__(key, value, set_func=self.__set_element)



class _VelocityArray(_IndexableArray):

    def __set_element(self, key, value):
        self.chain._pyb_client.resetJointState(
            self.chain._pyb_id,
            self.chain._Chain__nonfixed_indices_pyb[key],
            targetValue=self.chain._pyb_client.getJointState(self.chain._pyb_id, key)[0],
            targetVelocity=float(value),
        )

    def __setitem__(self, key, value):
        super().__setitem__(key, value, set_func=self.__set_element)



class _EffortArray(_IndexableArray):

    def __set_element(self, key, value):
        self.chain._effort[key] = value
        self.chain._pyb_client.setJointMotorControl2(
            self.chain._pyb_id, self.chain._Chain__nonfixed_indices_pyb[key], self.chain._pyb_client.TORQUE_CONTROL, force=float(value)
        )

    def __setitem__(self, key, value):
        super().__setitem__(key, value, set_func=self.__set_element)


class _JointDampingArray(_IndexableArray):

    def __set_element(self, key, value):
        self.chain._joint_damping[key] = value
        self.chain._pyb_client.changeDynamics(
            self.chain._pyb_id, self.chain._Chain__nonfixed_indices_pyb[key], jointDamping=float(value)
        )

    def __setitem__(self, key, value):
        super().__setitem__(key, value, set_func=self.__set_element)



class _JointLimitsArray(_IndexableArray):

    def __set_element(self, key, value):
        self.chain._joint_limits[key] = value
        self.chain._pyb_client.changeDynamics(
            self.chain._pyb_id, self.chain._Chain__nonfixed_indices_pyb[key], jointLowerLimit=float(value[0]), jointUpperLimit=float(value[1])
        )

    def __setitem__(self, key, value):
        super().__setitem__(key, value, set_func=self.__set_element)



class _JointEffortLimitsArray(_IndexableArray):

    def __set_element(self, key, value):
        self.chain._joint_effort_limits[key] = value
        self.chain._pyb_client.changeDynamics(
            self.chain._pyb_id, self.chain._Chain__nonfixed_indices_pyb[key], jointLimitForce=float(value)
        )

    def __setitem__(self, key, value):
        super().__setitem__(key, value, set_func=self.__set_element)



# http://wiki.ros.org/urdf/XML/link
# http://wiki.ros.org/urdf/XML/joint
def _chain_from_urdf(urdf_path: str):
    import pathlib
    base_dir = pathlib.Path(urdf_path).parent

    try:
       from collections.abc import Iterable
    except ImportError:
       from collections import Iterable
    import sys
    sys.modules['collections'].Iterable = Iterable
    
    from urdf_parser_py.urdf import URDF

    urdf = URDF.from_xml_string(open(urdf_path).read())

    G = nx.DiGraph()

    material_map = {}
    for material in urdf.materials:
        material_map[material.name] = {
            'color': np.array(material.color.rgba),
            'texture': material.texture,
        }

    for link in urdf.links:

        inertia_diagonal = np.zeros((3,))
        inertia_offset = np.eye(4)
        if link.inertial is not None:

            ixx = link.inertial.inertia.ixx
            ixy = link.inertial.inertia.ixy
            ixz = link.inertial.inertia.ixz
            iyy = link.inertial.inertia.iyy
            iyz = link.inertial.inertia.iyz
            izz = link.inertial.inertia.izz

            inertia = np.array([
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz]
            ])

            if link.inertial.origin is not None:
                R = _R.from_euler('xyz', link.inertial.origin.rotation).as_matrix()
                inertia_offset = trans_mat(
                    pos=np.array(link.inertial.origin.position), 
                    orn=R,
                )
                I_prime = R @ inertia @ R.T # I don't think this is necessary ¯\_(ツ)_/¯
            else:
                I_prime = inertia

            inertia_diagonal = np.sort(np.linalg.eigvals(I_prime).real)[::-1]

        coll_shape = phys.CollisionShape(Shape.EMPTY)
        coll_args = {}
        if (collision_shape := link.collisions[0] if len(link.collisions) > 0 else None):
            if collision_shape.origin is not None:
                offset = trans_mat(
                    pos=np.array(collision_shape.origin.position), 
                    orn=_R.from_euler('xyz', collision_shape.origin.rotation).as_matrix()
                )
            else:
                offset = np.eye(4)
            coll_args = {'offset': np.copy(offset)}

            if hasattr(collision_shape.geometry, 'size'):
                coll_type = Shape.CUBOID
                coll_args.update({'half_extents': np.array(collision_shape.geometry.size) / 2.0})
            elif hasattr(collision_shape.geometry, 'radius') and not hasattr(collision_shape.geometry, 'length'):
                coll_type = Shape.SPHERE
                coll_args.update({'radius': collision_shape.geometry.radius})
            elif hasattr(collision_shape.geometry, 'length') and hasattr(collision_shape.geometry, 'radius'):
                coll_type = Shape.CYLINDER
                coll_args.update({'radius': collision_shape.geometry.radius, 'height': collision_shape.geometry.length})
            elif hasattr(collision_shape.geometry, 'filename'):
                coll_type = Shape.MESH
                scale = np.array(collision_shape.geometry.scale) if collision_shape.geometry.scale is not None else np.ones((3,))
                coll_args.update({'filename': str(base_dir / collision_shape.geometry.filename), 'mesh_scale': scale})

            coll_shape = phys.CollisionShape(coll_type, **coll_args)

        vis_shape = gfx.VisualShape(Shape.EMPTY)
        vis_args = {}
        if (visual_shape := link.visuals[0] if len(link.visuals) > 0 else None):
            if visual_shape.origin is not None:
                offset = trans_mat(
                    pos=np.array(visual_shape.origin.position), 
                    orn=_R.from_euler('xyz', visual_shape.origin.rotation).as_matrix()
                )
            else:
                offset = np.eye(4)
            vis_args = {'offset': np.copy(offset)}

            if hasattr(visual_shape.geometry, 'size'):
                vis_type = Shape.CUBOID
                vis_args.update({'half_extents': np.array(visual_shape.geometry.size) / 2.0})
            elif hasattr(visual_shape.geometry, 'radius') and not hasattr(visual_shape.geometry, 'length'):
                vis_type = Shape.SPHERE
                vis_args.update({'radius': visual_shape.geometry.radius})
            elif hasattr(visual_shape.geometry, 'length') and hasattr(visual_shape.geometry, 'radius'):
                vis_type = Shape.CYLINDER
                vis_args.update({'radius': visual_shape.geometry.radius, 'height': visual_shape.geometry.length})
            elif hasattr(visual_shape.geometry, 'filename'):
                vis_type = Shape.MESH
                scale = np.array(visual_shape.geometry.scale) if visual_shape.geometry.scale is not None else np.ones((3,))
                vis_args.update({'filename': str(base_dir / visual_shape.geometry.filename), 'mesh_scale': scale})

            if material := visual_shape.material:
                if material.name in material_map:
                    vis_args.update({'color': material_map[material.name]['color']})
                    vis_args.update({'texture': material_map[material.name]['texture']})
                else:
                    vis_args.update({'color': np.array(material.color.rgba)})
                    vis_args.update({'texture': material.texture})

            vis_shape = gfx.VisualShape(vis_type, **vis_args)
                
        link_i = phys.Link(
            name=link.name,
            offset=inertia_offset,
            mass=link.inertial.mass if link.inertial is not None else 0,
            collision_shape=coll_shape,
            visual_shape=vis_shape,
            inertia_diagonal=inertia_diagonal,
        )
        
        G.add_node(link.name, link=link_i)

    for joint in urdf.joints:
        if joint.type == 'continuous':
            joint_type = phys.Constraint.REVOLUTE
        else:
            joint_type = getattr(phys.Constraint, joint.type.upper())
        
        if joint.origin is not None:
            offset = trans_mat(
                pos=np.array(joint.origin.position), 
                orn=_R.from_euler('xyz', joint.origin.rotation).as_matrix()
            )
        else:
            offset = np.eye(4)

        if joint.dynamics is not None:
            if joint.dynamics.damping is not None:
                joint_args.update({'damping': joint.dynamics.damping})

        if joint.axis is not None:
            axis = np.array(joint.axis)
        else:
            assert joint.type == 'fixed'
            axis = np.array([0, 0, 0])

        joint_args = {}
        if joint.limit is not None:
            if hasattr(joint.limit, 'lower'):
                joint_args.update({'lower_limit': joint.limit.lower})
            if hasattr(joint.limit, 'upper'):
                joint_args.update({'upper_limit': joint.limit.upper})
            if hasattr(joint.limit, 'velocity'):
                joint_args.update({'max_velocity': joint.limit.velocity})
            if hasattr(joint.limit, 'effort'):
                joint_args.update({'max_effort': joint.limit.effort})
                
        joint_i = phys.Joint(
            name=joint.name,
            offset=offset,
            axis=axis,
            type=joint_type,
            **joint_args,
        )
            
        G.add_edge(joint.parent, joint.child, joint=joint_i)

    try:
        delattr(sys.modules['collections'], 'Iterable')
    except:
        pass

    return phys.GraphChain(G)

