import contextlib
import tempfile

import numpy as np
import trimesh

from numbotics.math.spatial import trans_mat


# Originally the goal was to only load meshes using trimesh only if they were not obj files, 
# then convert them using trimesh. But, Meshcat does not have any easy way to modify the mesh, 
# so we will always load the mesh with trimesh and apply transformations, then pass the same 
# mesh into Meshcat and PyBullet.
# NOTE: PyBullet supports mesh scaling and convex decomposition, but we choose to forgo these features.

# This helper function will allow meshes of general types to be loaded, modified,
# then imported to both PyBullet and Meshcat without any additional processing.
@contextlib.contextmanager
def load_mesh(
    filename: str,
    mesh_scale: np.ndarray = np.array([1.0, 1.0, 1.0]),
    offset: np.ndarray = np.eye(4),
    convex_decomposition: bool = False,
    auto_center: bool = False,
    **kwargs,
):
    mesh = trimesh.load(filename)
    if auto_center:
        mesh.apply_translation(-mesh.center_mass)
    mesh.apply_transform(trans_mat(orn=np.diag(mesh_scale)))
    mesh.apply_transform(offset)
    if convex_decomposition:
        convex_parts = mesh.convex_decomposition()
        mesh = trimesh.Scene(convex_parts)
    with tempfile.NamedTemporaryFile(suffix=".obj", delete=True) as tmp_file:
        mesh.export(tmp_file.name)
        yield tmp_file.name
        