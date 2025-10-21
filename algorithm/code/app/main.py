import time
import os
import tempfile
import numpy as np
import pyvista as pv

# Import our new modules
import data_loader
import volume_processor
import mesh_processor

def get_user_params():
    """Gathers all necessary parameters from the user via the command line."""
    params = {}
    print("--- T1MESCULPTURES Setup ---")
    
    # Data Loader Params
    params['path'] = input("Enter the path to your image sequence folder: ")
    params['filetype'] = input("Enter the file type (e.g., png): ")
    params['threshhold'] = int(input("Enter the binary threshold (0-255, e.g., 127): "))
    
    # Volume Processor Params
    params['fps'] = int(input("Enter the FPS of the source animation: "))
    params['scale_factor'] = float(input("Enter the downsampling scale factor (e.g., 0.3): "))
    
    # Smoothing Params
    params['smooth_method'] = input("Choose smoothing method ['gaussian', 'constrained', 'none']: ").lower()
    params['smooth_kwargs'] = {}
    if params['smooth_method'] == 'gaussian':
        sigma = float(input("Enter Gaussian sigma (e.g., 1.5): "))
        params['smooth_kwargs']['sigma'] = sigma
    elif params['smooth_method'] == 'constrained':
        iters = int(input("Enter constrained max iterations (e.g., 50): "))
        params['smooth_kwargs']['max_iters'] = iters
        
    # Output Params
    params['output_name'] = input("Enter the base name for the output .stl files: ")
    
    return params

def main():
    """Main execution function to run the full pipeline."""
    
    total_start_time = time.time()
    
    # --- Create unique temporary directory for session ---
    temp_dir = tempfile.mkdtemp(prefix="t1mesculptures_")
    print(f"Caching intermediate files in: {temp_dir}")
    #  Define cache file paths
    volume_cache_path = os.path.join(temp_dir, "smoothed_volume.npy")
    mesh_cache_path = os.path.join(temp_dir, "raw_mesh.ply")
    
    # --- Get all parameters from the user ---
    try:
        params = get_user_params()
    except ValueError as e:
        print(f"Invalid input: {e}. Please try again.")
        return

    # --- DATA LOADING ---
    # Fast, no cache needed
    print("\n" + "="*30)
    print("STAGE 1: LOADING DATA")
    print("="*30)
    stage_time = time.time()
    
    frames = data_loader.load_frames(
        params['path'], 
        params['filetype'], 
        params['threshhold']
    )
    
    if not frames:
        print("Failed to load frames. Exiting.")
        return
        
    print(f"--- Stage 1 finished in {time.time() - stage_time:.2f} seconds ---")

    # --- VOLUME PROCESSING ---
    print("\n" + "="*30)
    print("STAGE 2: PROCESSING VOLUME")
    print("="*30)
    stage_time = time.time()
    
    pointcloud, totalFrames, scaled_height, scaled_width = volume_processor.create_pointcloud_scaffold(
        frames, 
        params['scale_factor']
    )
    
    pointcloud = volume_processor.fill_pointcloud(
        pointcloud, 
        frames, 
        scaled_height, 
        scaled_width
    )

    smoothed_volume = None
    
    # rescaled_volume = volume_processor.scale_volume(
    #     pointcloud, 
    #     totalFrames, 
    #     params['fps'], 
    #     scaled_height, 
    #     scaled_width
    # )
    
    # smoothed_volume = volume_processor.smooth_volume(
    #     rescaled_volume, 
    #     params['smooth_method'], 
    #     **params['smooth_kwargs']
    # )

    # Caching the Volume
    if os.path.exists(volume_cache_path):
        print("Loading cached volume from disk...")
        smoothed_volume = np.load(volume_cache_path)
    else:
        print("No cache found. Running volume processing...")
        rescaled_volume = volume_processor.scale_volume(
            pointcloud, 
            totalFrames, 
            params['fps'], 
            scaled_height, 
            scaled_width
        )
        smoothed_volume = volume_processor.smooth_volume(
            rescaled_volume, 
            params['smooth_method'], 
            **params['smooth_kwargs']
        )
        
        np.save(volume_cache_path, smoothed_volume)
        print(f"Saved volume to cache: {volume_cache_path}")
    print(f"--- Stage 2 finished in {time.time() - stage_time:.2f} seconds ---")

    # --- MESH EXTRACTION ---
    print("\n" + "="*30)
    print("STAGE 3: EXTRACTING MESH")
    print("="*30)
    stage_time = time.time()
    
    original_surface = None
    vertices = None
    faces = None    
    # Caching the Mesh
    if os.path.exists(mesh_cache_path):
        print("Loading cached raw mesh from disk...")
        # We don't need vertices/faces, just the PyVista object
        original_surface = pv.read(mesh_cache_path)
        # We need to get the vertices/faces back if optimize_mesh needs them
        # (This logic can be cleaned up, but it's the main idea)
        vertices = original_surface.points
        faces = original_surface.faces.reshape(-1, 4)[:, 1:] 
    else:
        print("No cache found. Running mesh extraction...")
        vertices, faces = mesh_processor.extract_mesh(smoothed_volume)
        
        # Save the result to our cache!
        original_surface = pv.PolyData(vertices.astype(np.float32), np.hstack((np.full((faces.shape[0], 1), 3), faces)))
        original_surface.save(mesh_cache_path)
        print(f"Saved raw mesh to cache: {mesh_cache_path}")
    # vertices, faces = mesh_processor.extract_mesh(smoothed_volume)
    
    # Save the original, un-decimated mesh
    try:
        print("Saving original (un-decimated) mesh...")
        original_surface = pv.PolyData(vertices.astype(np.float32), np.hstack((np.full((faces.shape[0], 1), 3), faces)))
        original_surface.save(f"{params['output_name']}_original.stl")
        print(f"Saved to {params['output_name']}_original.stl")
    except Exception as e:
        print(f"Could not save original mesh: {e}")
        
    print(f"--- Stage 3 finished in {time.time() - stage_time:.2f} seconds ---")

    # --- INTERACTIVE OPTIMIZATION ---
    print("\n" + "="*30)
    print("STAGE 4: OPTIMIZING MESH")
    print("="*30)
    
    while True:
        try:
            reduction = float(input("Enter target decimation (e.g., 0.9 for 90%), 0 for none, or a negative number to exit: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if reduction < 0:
            print("Exiting interactive loop.")
            break
            
        stage_time = time.time()
        
        final_surface = mesh_processor.optimize_mesh(
            vertices, 
            faces, 
            reduction
        )
        
        print(f"--- Optimization/Repair finished in {time.time() - stage_time:.2f} seconds ---")
        
        if final_surface:
            print("Displaying final mesh. Close the window to continue.")
            plotter = pv.Plotter()
            plotter.add_mesh(final_surface, show_edges=True)
            plotter.show()

            satisfied = input("Are you satisfied with this version? (y/n): ").lower()
            if satisfied == 'y':
                final_filename = f"{params['output_name']}_{int(reduction*100)}percent_final.stl"
                final_surface.save(final_filename)
                print(f"Final mesh saved to '{final_filename}'.")
                break
            else:
                print("Let's try a different reduction value.")
        else:
            print("Mesh optimization failed.")
            break

    print(f"\n--- Total Process Complete in {time.time() - total_start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()