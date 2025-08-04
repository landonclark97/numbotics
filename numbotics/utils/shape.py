from enum import Enum, auto

import numpy as np
from scipy.spatial.transform import Rotation as _R
from meshcat.geometry import (
    Box,
    Sphere,
    Cylinder,
    ObjMeshGeometry,
    Plane,
)

from .mesh import load_mesh



class Shape(Enum):
    CUBE = auto()
    CUBOID = auto()
    SPHERE = auto()
    CYLINDER = auto()
    CAPSULE = auto()
    MESH = auto()
    PLANE = auto()
    EMPTY = auto()
    
    def create_visual_shape(self, **kwargs):
        if self == Shape.CUBE or self == Shape.CUBOID:
            half_extents = kwargs['half_extents']
            return Box(lengths=(2.0 * half_extents).tolist())
        elif self == Shape.SPHERE:
            return Sphere(radius=kwargs['radius'])
        elif self == Shape.MESH:
            # No need to perform convex decomposition for visual shapes
            _ = kwargs.pop('convex_decomposition', None)
            filename = kwargs.pop('filename')
            try:
                with load_mesh(filename=filename, **kwargs) as tmp_file:
                    return ObjMeshGeometry.from_file(tmp_file)
            except Exception as e:
                raise ValueError(f"Invalid mesh file: {filename}") from e
        elif self == Shape.CYLINDER:
            return Cylinder(height=kwargs['height'], radius=kwargs['radius'])
        elif self == Shape.PLANE:
            return Plane(width=1e5, height=1e5, widthSegments=10, heightSegments=10)
        elif self == Shape.CAPSULE:
            raise NotImplementedError("Capsule visual shape not implemented")
        elif self == Shape.EMPTY:
            return None
        else:
            raise ValueError(f"Unknown visual shape type: {self}")
        
    def register_collision_shape(self, **kwargs):
        import numbotics.physics as phys
        pyb = kwargs.pop('pyb_client', None)
        if pyb is None:
            pyb = phys.get_world(name=kwargs.pop('world_name', None))._pyb_client
        offset = kwargs.pop('offset', np.eye(4))
        if self == Shape.CUBE or self == Shape.CUBOID:
            col_id = pyb.createCollisionShape(
                shapeType=pyb.GEOM_BOX, 
                halfExtents=kwargs['half_extents'].tolist(),
                collisionFramePosition=offset[:3, 3].tolist(),
                collisionFrameOrientation=_R.from_matrix(offset[:3, :3]).as_quat().tolist(),
            )
        elif self == Shape.SPHERE:
            col_id = pyb.createCollisionShape(
                shapeType=pyb.GEOM_SPHERE, 
                radius=kwargs['radius'],
                collisionFramePosition=offset[:3, 3].tolist(),
                collisionFrameOrientation=_R.from_matrix(offset[:3, :3]).as_quat().tolist(),
            )
        elif self == Shape.CYLINDER:
            col_id = pyb.createCollisionShape(
                shapeType=pyb.GEOM_CYLINDER, 
                radius=kwargs['radius'], 
                height=kwargs['height'],
                collisionFramePosition=offset[:3, 3].tolist(),
                collisionFrameOrientation=_R.from_matrix(offset[:3, :3]).as_quat().tolist(),
            )
        elif self == Shape.MESH:
            filename = kwargs.pop('filename')
            try:
                with load_mesh(
                    filename=filename,
                    offset=offset,
                    **kwargs
                ) as tmp_file:
                    col_id = pyb.createCollisionShape(
                        shapeType=pyb.GEOM_MESH, 
                        fileName=tmp_file,
                    )
            except Exception as e:
                raise ValueError(f"Invalid mesh file: {filename}") from e
        elif self == Shape.CAPSULE:
            col_id = pyb.createCollisionShape(
                shapeType=pyb.GEOM_CAPSULE, 
                radius=kwargs['radius'], 
                height=kwargs['height'],
                collisionFramePosition=offset[:3, 3].tolist(),
                collisionFrameOrientation=_R.from_matrix(offset[:3, :3]).as_quat().tolist(),
            )
        elif self == Shape.PLANE:
            col_id = pyb.createCollisionShape(
                shapeType=pyb.GEOM_PLANE,
                planeNormal=kwargs['normal'].tolist(),
                collisionFramePosition=offset[:3, 3].tolist(),
                collisionFrameOrientation=_R.from_matrix(offset[:3, :3]).as_quat().tolist(),
            )
        elif self == Shape.EMPTY:
            col_id = -1
        else:
            raise ValueError(f'Invalid geometry: {self}')
        return col_id



__shape_kwargs = {
    'offset',
    'half_extents',
    'radius',
    'height',
    'width',
    'normal',
    'filename',
    'color',
    'mesh_scale',
    'auto_center',
    'convex_decomposition',
}
def parse_shape_kwargs(kwargs: dict) -> dict:
    shape_info = {}
    for key in list(kwargs.keys()):
        if key in __shape_kwargs:
            shape_info[key] = kwargs.pop(key)
    return kwargs, shape_info