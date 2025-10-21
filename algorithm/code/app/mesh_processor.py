# Handles all surface mesh operations

import mcubes as mc
import numpy as np
import pyvista as pv
import pymeshfix as mf

def extract_mesh(volume):
    vertices, faces = mc.marching_cubes(volume, 0)
    print(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces.")
    return vertices, faces

def optimize_mesh(vertices, faces, reduction_factor):
    # --- Create the PyVista mesh object ---
    faces_pyvista = np.hstack((np.full((faces.shape[0], 1), 3), faces))
    smooth_surface = pv.PolyData(vertices.astype(np.float32), faces_pyvista)

    # --- Perform decimation logic ---
    if reduction_factor == 0:
        print("Skipping decimation. Using the un-decimated mesh.")
        final_surface = smooth_surface
    else:
        # Perform decimation only if the value is greater than 0
        print(f"Decimating the smooth mesh by {reduction_factor * 100}%...")
        final_surface = smooth_surface.decimate_pro(reduction=reduction_factor)

    # --- Verification and Repair ---
    is_manifold = final_surface.is_manifold
    print(f"\n--- Result ---")
    print(f"Final mesh has {final_surface.n_points} vertices and {final_surface.n_cells} faces.")
    print(f"Is the mesh manifold? {is_manifold}")

    if not is_manifold:
        print("WARNING: The resulting mesh is not manifold. Attempting repair with PyMeshFix...")
        final_surface = repair_and_check_mesh(final_surface)

    return final_surface
            
def repair_and_check_mesh(mesh):
    # Create a MeshFix object from the PyVista data
    meshfix = mf.MeshFix(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    
    # Run the repair
    meshfix.repair()

    # 1. Get the repaired vertices and faces from the MeshFix object
    repaired_verts = meshfix.v
    repaired_faces = meshfix.f

    # 2. Convert the faces to the PyVista format
    repaired_faces_pyvista = np.hstack((np.full((repaired_faces.shape[0], 1), 3), repaired_faces))    
    
    # 3. Create a new PyVista PolyData object from the repaired data
    repaired_mesh = pv.PolyData(repaired_verts, repaired_faces_pyvista)
    
    print(f"Is repaired mesh manifold? {repaired_mesh.is_manifold} ✅")
    mesh = repaired_mesh
    return mesh