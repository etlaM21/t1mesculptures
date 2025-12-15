# --- T1MESCULPTURES - Interactive Mesh Generation Script ---

# Step 1: Import necessary libraries
import cv2 as cv
import numpy as np
import os
import MarchingNumPy.MarchingCubesLorensen as MarchingNumPy
from stl import mesh
import pyvista as pv  # Import PyVista for mesh simplification and viewing

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

# Step 2: Get user input for key parameters
print("--- T1MESCULPTURES Initial Setup ---")
path_str = input("Enter the path to your image sequence folder: ")
path = os.fsencode(path_str)
FPS = int(input("Enter the FPS of the source animation: "))
output_name = input("Enter the base name for the output .stl files: ")

# a) Ask for user input for pre-processing (downsampling)
scale_factor = float(input("Enter the downsampling scale factor (e.g., 0.5 for 50% resolution): "))

# --- Image Loading and Processing ---
frames = []
try:
    with os.scandir(path) as it:
        # Sort entries to ensure frames are in the correct order
        sorted_entries = sorted(it, key=lambda entry: entry.name)
        for entry in sorted_entries:
            filename = os.fsdecode(entry.path)
            if filename.endswith(".png") and entry.is_file():
                frames.append(Frame(filename))
except FileNotFoundError:
    print(f"Error: The directory '{path_str}' was not found. Please check the path and try again.")
    exit()

if not frames:
    print("Error: No .png files were found in the specified directory.")
    exit()

totalFrames = len(frames)
print(f"\nSuccessfully loaded {totalFrames} frames at {FPS} FPS.")

# --- Point Cloud Generation (with Downsampling) ---
print("--- Generating 3D Point Cloud ---")
# Calculate the new, smaller dimensions based on the scale factor
scaled_width = int(frames[0].width * scale_factor)
scaled_height = int(frames[0].height * scale_factor)
print(f"Downsampling images from ({frames[0].width}, {frames[0].height}) to ({scaled_width}, {scaled_height}).")

# Pre-allocate the 3D numpy array (our volume) with the new dimensions
pointcloud = np.zeros((scaled_width, scaled_height, totalFrames))

# Populate the volume with the downsampled image masks
for i, frame in enumerate(frames):
    # Resize the mask using the scale factor. INTER_NEAREST is fast and good for binary images.
    resized_mask = cv.resize(frame.mask, (scaled_width, scaled_height), interpolation=cv.INTER_NEAREST)
    # Assign the 2D slice to its position in the 3D volume
    pointcloud[:, :, i] = resized_mask

print(f"Pointcloud generated with shape {pointcloud.shape}.")

# --- Marching Cubes ---
print("--- Applying Marching Cubes Algorithm ---")
level = THRESHHOLD  # The value that represents the surface boundary

# Generate the high-resolution vertices and faces from the volume
vertices, faces = MarchingNumPy.marching_cubes_lorensen(np.pad(pointcloud, 1), level=level)
print(f"Generated initial mesh with {len(vertices)} vertices and {len(faces)} faces.")

# --- Z-Axis Rescaling (Time Dimension) ---
print("--- Rescaling Z-Axis ---")
# Calculate the final physical height of the sculpture based on your formula
maxHeight = (totalFrames / FPS) * max(scaled_width, scaled_height)
print(f"Rescaling output with a maximum height of {maxHeight:.2f} units.")
# Apply the scaling to the Z-coordinate of each vertex
vertices[:, 2] = (vertices[:, 2] / totalFrames) * maxHeight

# --- Save the Original, Un-simplified Mesh ---
# b) Store the mesh before using PyVista for comparison or other uses.
original_filename = f"{output_name}_original.stl"
original_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        original_mesh.vectors[i][j] = vertices[f[j], :]
original_mesh.save(original_filename)
print(f"Saved the original, un-simplified mesh to '{original_filename}'.")

# --- Interactive Simplification Loop ---
# b) Create a loop to let the user find the best simplification target.
print("\n--- Interactive Mesh Simplification ---")

# First, convert our mesh data into a PyVista-compatible format
# PyVista's format is [num_points, point_idx_1, point_idx_2, ...], so for triangles it's [3, v0, v1, v2]
faces_pyvista = np.hstack((np.full((faces.shape[0], 1), 3), faces))
# Convert vertices to a PyVista-compatible data type (float32)
vertices = vertices.astype(np.float32)
original_surface = pv.PolyData(vertices, faces_pyvista)

while True:
    # a) Ask for user input for the reduction target
    try:
        target_reduction = float(input("\nEnter target reduction (e.g., 0.9 for 90% reduction, or 0 to exit): "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        continue

    if target_reduction <= 0 or target_reduction >= 1:
        print("Exiting interactive loop.")
        break

    # Apply the mesh simplification (decimation)
    print(f"Simplifying mesh with a {target_reduction*100}% target reduction...")
    simplified_surface = original_surface.decimate_pro(reduction=target_reduction)
    
    print(f"Simplified mesh has {simplified_surface.n_points} vertices and {simplified_surface.n_cells} cells (faces).")
    
    # b) Show the resulting mesh in an interactive 3D window
    print("Displaying simplified mesh. Close the window to continue.")
    plotter = pv.Plotter()
    plotter.add_mesh(simplified_surface, show_edges=True)
    plotter.show() # This will open a window with the 3D model

    # Ask the user if they are happy with the result
    satisfied = input("Are you satisfied with this version? (y/n): ").lower()
    if satisfied == 'y':
        # If yes, save this version and exit the loop
        final_filename = f"{output_name}_simplified_{int(target_reduction*100)}percent.stl"
        simplified_surface.save(final_filename)
        print(f"Final mesh saved to '{final_filename}'.")
        break
    else:
        print("Let's try a different reduction value.")

print("\n--- Process Complete ---")