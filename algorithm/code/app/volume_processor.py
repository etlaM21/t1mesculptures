# Creates Pointcloud from Frames
# Scales and Smoothes the Pointcloud / Volume

import numpy as np
import cv2 as cv
from scipy.ndimage import zoom
import mcubes as mc

def create_pointcloud_scaffold(frames, scale_factor):
    if not frames:
        print("Cannot create pointcloud: no frames loaded.")
        return None
    
    print("--- Generating and Rescaling 3D Point Cloud ---")
    scaled_width = int(frames[0].width * scale_factor)
    scaled_height = int(frames[0].height * scale_factor)
    print(f"Downsampling images to ({scaled_width}, {scaled_height}).")

    # Create the initial, unscaled pointcloud
    totalFrames = len(frames)
    pointcloud = np.zeros((totalFrames, scaled_height, scaled_width), dtype=np.float32) # Data Type float32 to end up with less giant volume after zoom()
    return pointcloud, totalFrames, scaled_height, scaled_width



def fill_pointcloud(pointcloud, frames, scaled_height, scaled_width):
    if not frames:
        print("Cannot fill pointcloud: no frames loaded.")
        return None
    
    if pointcloud is None:
         print("Cannot fill pointcloud: scaffold is None.")
         return None
    
    print("Filling pointcloud...")
    
    for i, frame in enumerate(frames): 
        resized_mask = cv.resize(frame.mask, (scaled_width, scaled_height), interpolation=cv.INTER_NEAREST)
        pointcloud[i, :, :] = resized_mask / 255.0

    # Add 1 layer of padding with value 0 (representing 'outside')
    # around all three axes (time, height, width).
    # So marching cubes properly encloses the mesh
    print("Adding 2-voxel padding around the volume...")
    padded_pointcloud = np.pad(pointcloud, pad_width=2, mode='constant', constant_values=0)

    return padded_pointcloud


def scale_volume(pointcloud, totalFrames, fps, scaled_height, scaled_width):
    # Calculate the desired height of the Z-axis in pixels
    maxHeight = (totalFrames / fps) * max(scaled_width, scaled_height)
    # Calculate the zoom factor for the Z-axis
    z_zoom_factor = maxHeight / totalFrames
    print(f"Stretching Z-axis by a factor of {z_zoom_factor:.2f}...")

    # Use scipy's zoom to resample the volume to the correct proportions
    # This creates a new volume where each voxel is roughly a cube
    rescaled_volume = zoom(pointcloud, (z_zoom_factor, 1, 1), order=1)
    return rescaled_volume


def smooth_volume(volume, method, **kwargs):
    if volume is None:
        print("Cannot smooth volume: volume is None.")
        return None
        
    print(f"\n--- Smoothing the 3D volume using '{method}' method ---")

    # --- NEW: Interactive and Correct Smoothing Control ---
    # Based on the official source code for mcubes.smooth()
    
    if method in ['gaussian', 'constrained']:
        print(f"Applying '{method}' smoothing with params: {kwargs}...")
        smoothed_volume = mc.smooth(volume, method=method, **kwargs)
    else:
        print("No valid smoothing method chosen or method is 'none'. Skipping smoothing.")
        smoothed_volume = volume

    return smoothed_volume