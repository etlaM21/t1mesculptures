# --- T1MESCULPTURES - Final Version with PyMCubes ---

# Import necessary libraries
import cv2 as cv
import numpy as np
import os
import time
import pyvista as pv
import pymeshfix as mf
import mcubes as mc
from stl import mesh
from scipy.ndimage import zoom

# --- Global Settings ---
THRESHHOLD = 127


# --- Frame Class (for image processing) ---
# This class handles loading and processing each individual image file.
class Frame:
    def __init__(self, filepath):
        self.threshhold = THRESHHOLD
        self.filepath = filepath
        self.image = cv.imread(filepath)
        self.height, self.width, self.channels = self.image.shape
        # Convert the image to grayscale immediately upon creation
        self.imagegray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        # Create the binary mask from the grayscale image
        self.mask = self.getThreshhold()

    def getThreshhold(self):
        # This method creates a simple black and white (binary) image
        # based on the THRESHHOLD value.
        _, img_treshhold = cv.threshold(self.imagegray, self.threshhold, 255, cv.THRESH_BINARY)
        return img_treshhold

# --- Main Script Logic ---
print("--- T1MESCULPTURES Initial Setup ---")
path_str = input("Enter the path to your image sequence folder: ")
path = os.fsencode(path_str)
FPS = int(input("Enter the FPS of the source animation: "))
output_name = input("Enter the base name for the output .stl files: ")
scale_factor = float(input("Enter the downsampling scale factor (e.g., 0.5): "))

# --- Image Loading and Processing ---
frames = []
try:
    with os.scandir(path) as it:
        sorted_entries = sorted(it, key=lambda entry: entry.name)
        for entry in sorted_entries:
            filename = os.fsdecode(entry.path)
            if filename.endswith(".png") and entry.is_file():
                frames.append(Frame(filename))
except FileNotFoundError:
    print(f"Error: The directory '{path_str}' was not found.")
    exit()

if not frames:
    print("Error: No .png files were found in the specified directory.")
    exit()

totalFrames = len(frames)
print(f"\nSuccessfully loaded {totalFrames} frames.")

# --- Point Cloud Generation and Rescaling ---
print("--- Generating and Rescaling 3D Point Cloud ---")
scaled_width = int(frames[0].width * scale_factor)
scaled_height = int(frames[0].height * scale_factor)
print(f"Downsampling images to ({scaled_width}, {scaled_height}).")

# Create the initial, unscaled pointcloud
pointcloud = np.zeros((totalFrames, scaled_height, scaled_width))
for i, frame in enumerate(frames):
    resized_mask = cv.resize(frame.mask, (scaled_width, scaled_height), interpolation=cv.INTER_NEAREST)
    pointcloud[i, :, :] = resized_mask / 255.0

# ** NEW: Correct Z-Axis Scaling of the Volume **
# Calculate the desired height of the Z-axis in pixels
maxHeight = (totalFrames / FPS) * max(scaled_width, scaled_height)
# Calculate the zoom factor for the Z-axis
z_zoom_factor = maxHeight / totalFrames
print(f"Stretching Z-axis by a factor of {z_zoom_factor:.2f}...")

# Use scipy's zoom to resample the volume to the correct proportions
# This creates a new volume where each voxel is roughly a cube
rescaled_volume = zoom(pointcloud, (z_zoom_factor, 1, 1), order=1)

# --- Smooth the Volume and Extract Mesh with mcubes ---
print("\n--- Smoothing the 3D volume and generating mesh ---")

# --- NEW: Interactive and Correct Smoothing Control ---
# Based on the official source code for mcubes.smooth()

# Initialize an empty dictionary for keyword arguments
kwargs = {}

# Ask the user to choose the method
smoothing_method = input("Choose smoothing method ['gaussian', 'constrained']: ").lower()

if smoothing_method == 'gaussian':
    # If gaussian, ask for the sigma value
    sigma_val = float(input("Enter Gaussian sigma (e.g., 1.0 is subtle, 3.0 is strong): "))
    kwargs['sigma'] = sigma_val
    
elif smoothing_method == 'constrained':
    # If constrained, ask for the max_iters value
    max_iters_val = int(input("Enter constrained max iterations (e.g., 50 is gentle, 500 is strong): "))
    kwargs['max_iters'] = max_iters_val

# Check if a valid method was chosen, otherwise, skip smoothing
if smoothing_method in ['gaussian', 'constrained']:
    print(f"Applying '{smoothing_method}' smoothing...")
    start_time = time.time()
    smoothed_volume = mc.smooth(rescaled_volume, method=smoothing_method, **kwargs)
    print(f"--> Smoothing finished in {time.time() - start_time:.2f} seconds.")
else:
    print("No valid smoothing method chosen. Skipping smoothing.")
    smoothed_volume = rescaled_volume

# Extract the mesh from the (potentially) smoothed volume
# The isovalue is 0, as specified in the docstring for a smoothed volume.
print("Running marching cubes...")
start_time = time.time()
vertices, faces = mc.marching_cubes(smoothed_volume, 0)

print(f"PyMCubes finished in {time.time() - start_time:.2f} seconds.")
print(f"Generated mesh with {len(vertices)} vertices and {len(faces)} faces.")

# --- Save the Original, Un-simplified Mesh ---
print("--- Saving Original Mesh ---")
original_filename = f"{output_name}_original.stl"
original_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        original_mesh.vectors[i][j] = vertices[f[j], :]
original_mesh.save(original_filename)
print(f"Saved the original, un-simplified mesh to '{original_filename}'.")


# --- Final Decimation and Interactive Loop ---
print("\n--- Final Optimization ---")
# Convert to PyVista for final processing
faces_pyvista = np.hstack((np.full((faces.shape[0], 1), 3), faces))
smooth_surface = pv.PolyData(vertices.astype(np.float32), faces_pyvista)

while True:
    try:
        final_reduction = float(input("Enter target decimation (e.g., 0.9 for 90%), 0 for none, or a negative number to exit: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        continue

    # Exit only on a negative number
    if final_reduction < 0:
        print("Exiting interactive loop.")
        break

    # If the user enters 0, we use the original smoothed surface
    if final_reduction == 0:
        print("Skipping decimation. Using the un-decimated mesh.")
        final_surface = smooth_surface
    else:
        # Perform decimation only if the value is greater than 0
        print(f"Decimating the smooth mesh by {final_reduction * 100}%...")
        start_time = time.time()
        final_surface = smooth_surface.decimate_pro(reduction=final_reduction)
        print(f"--> Final decimation finished in {time.time() - start_time:.2f} seconds.")

    # --- Verification and Repair (No changes from here) ---
    is_manifold = final_surface.is_manifold
    print(f"\n--- Result ---")
    print(f"Final mesh has {final_surface.n_points} vertices and {final_surface.n_cells} faces.")
    print(f"Is the mesh manifold? {is_manifold}")

    if not is_manifold:
        print("WARNING: The resulting mesh is not manifold. Attempting repair with PyMeshFix...")
        
        # Create a MeshFix object from the PyVista data
        meshfix = mf.MeshFix(final_surface.points, final_surface.faces.reshape(-1, 4)[:, 1:])
        
        # Run the repair
        meshfix.repair()

        # 1. Get the repaired vertices and faces from the MeshFix object
        repaired_verts = meshfix.v
        repaired_faces = meshfix.f

        # 2. Convert the faces to the PyVista format
        repaired_faces_pyvista = np.hstack((np.full((repaired_faces.shape[0], 1), repaired_faces.shape[1]), repaired_faces))
        
        # 3. Create a new PyVista PolyData object from the repaired data
        repaired_mesh = pv.PolyData(repaired_verts, repaired_faces_pyvista)
        
        print(f"Is repaired mesh manifold? {repaired_mesh.is_manifold} ✅")
        final_surface = repaired_mesh

    print("Displaying final mesh. Close the window to continue.")
    plotter = pv.Plotter()
    plotter.add_mesh(final_surface, show_edges=True)
    plotter.show()

    satisfied = input("Are you satisfied with this version? (y/n): ").lower()
    if satisfied == 'y':
        final_filename = f"{output_name}_final.stl"
        final_surface.save(final_filename)
        print(f"Final mesh saved to '{final_filename}'.")
        break
    else:
        print("Let's try a different reduction value.")

print("\n--- Process Complete ---")