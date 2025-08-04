import open3d as o3d

import glob


stl_files = [f for f in glob.glob('./meshes/visual/*.dae')]
print(stl_files)

for f in stl_files:
    mesh = o3d.io.read_triangle_mesh(f)
    o3d.io.write_triangle_mesh(f.replace('dae', 'obj', 1), mesh)
