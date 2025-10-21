import os
import time
import threading  # For running the main process in the background
import numpy as np
import pyvista as pv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sv_ttk # Theme Sun-Valley
import sys # For redirecting stdout
# For video preview
from PIL import Image, ImageTk  
import cv2 as cv

# Import own modules
import data_loader
import volume_processor
import mesh_processor
import app_utils

class TextLogger:
    """A helper class to redirect stdout to a Tkinter Text widget."""
    def __init__(self, text_widget):
        self.widget = text_widget

    def write(self, msg):
        self.widget.configure(state="normal")  # Enable widget to insert text
        self.widget.insert(tk.END, msg)
        self.widget.see(tk.END)  # Auto-scroll to the end
        self.widget.configure(state="disabled") # Disable to make it read-only

    def flush(self):
        """Needed for stdout compatibility."""
        pass

class T1mesculpturesApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # --- Basic App Setup ---
        self.title("T1MESCULPTURES")
        self.geometry("1100x750")
        
        # --- App State ---
        self.frames_list = [] # Holds the loaded image frames for the video
        self.is_video_playing = False
        self.elapsed_at_pause = 0.0 # Tracks time for pause/resume -> Used for TIME-BASED animation playback instead of frame-based

        self.raw_mesh_cache_path = None
        self.final_mesh_result = None

        # --- Parameter Storage ---
        # Tkinter variables to auto-update widgets
        self.vars = {
            "path": tk.StringVar(),
            "filetype": tk.StringVar(value="png"),
            "output_name": tk.StringVar(value="t1mesculpture"),
            "threshold": tk.IntVar(value=127),
            "fps": tk.IntVar(value=30),
            "scale_factor": tk.DoubleVar(value=0.5),
            "smooth_method": tk.StringVar(value="constrained"),
            "smooth_sigma": tk.DoubleVar(value=1.5),
            "smooth_iters": tk.IntVar(value=50),
            "decimation": tk.DoubleVar(value=0.9),
        }

        # --- Main Layout ---
        # Two main columns
        self.controls_frame = ttk.Frame(self, width=400, padding=10)
        self.output_frame = ttk.Frame(self, padding=10)
        
        # Use grid layout for the main window
        self.columnconfigure(0, weight=0, minsize=400) # Control panel column
        self.columnconfigure(1, weight=1) # Output panel column
        self.rowconfigure(0, weight=1)
        
        self.controls_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.output_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        # --- Build the Widgets ---
        self.create_controls_widgets()
        self.create_output_widgets()
        
        # Set initial state for dynamic widgets
        self.update_smoothing_controls()

    def create_controls_widgets(self):
        """Populates the left-hand controls panel."""
        frame = self.controls_frame

        frame.rowconfigure(5, weight=1) # Last Row of column (under action frame) is the only one that resizes, meaning everything keeps sticking up top
        
        # --- 1. Input Frame ---
        input_frame = ttk.LabelFrame(frame, text="Input / Output", padding=10)
        input_frame.grid(row=0, column=0, sticky="nsew", pady=5)
        input_frame.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Image Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.vars["path"]).grid(row=1, column=0, columnspan=2, sticky="ew", padx=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_folder).grid(row=1, column=2, sticky="e", padx=5)

        ttk.Label(input_frame, text="File Type:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.vars["filetype"], width=10).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(input_frame, text="Output Name:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.vars["output_name"]).grid(row=3, column=1, columnspan=2, sticky="ew", padx=5)

        # --- 2. Data Frame ---
        data_frame = ttk.LabelFrame(frame, text="Data Parameters", padding=10)
        data_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        data_frame.columnconfigure(1, weight=1)
        
        self.create_slider(data_frame, "Threshold:", self.vars["threshold"], 0, 255, 0)
        self.create_slider(data_frame, "FPS:", self.vars["fps"], 1, 120, 1)
        self.create_slider(data_frame, "Scale Factor:", self.vars["scale_factor"], 0.1, 1.0, 2)

        # --- 3. Smoothing Frame ---
        smooth_frame = ttk.LabelFrame(frame, text="Smoothing", padding=10)
        smooth_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        smooth_frame.columnconfigure(1, weight=1)

        # Radio buttons
        ttk.Radiobutton(smooth_frame, text="None", variable=self.vars["smooth_method"], value="none", command=self.update_smoothing_controls).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(smooth_frame, text="Gaussian", variable=self.vars["smooth_method"], value="gaussian", command=self.update_smoothing_controls).grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(smooth_frame, text="Constrained", variable=self.vars["smooth_method"], value="constrained", command=self.update_smoothing_controls).grid(row=2, column=0, sticky="w")
        
        # Dynamic Sliders (we create both and hide/show them)
        self.sigma_frame = ttk.Frame(smooth_frame)
        self.sigma_frame.grid(row=1, column=1, sticky="ew")
        self.create_slider(self.sigma_frame, "Sigma:", self.vars["smooth_sigma"], 0.1, 5.0, 0)
        
        self.iters_frame = ttk.Frame(smooth_frame)
        self.iters_frame.grid(row=2, column=1, sticky="ew")
        self.create_slider(self.iters_frame, "Iterations:", self.vars["smooth_iters"], 10, 500, 0)

        # --- 4. Optimization Frame ---
        opt_frame = ttk.LabelFrame(frame, text="Optimization", padding=10)
        opt_frame.grid(row=3, column=0, sticky="nsew", pady=5)
        opt_frame.columnconfigure(1, weight=1)
        self.create_slider(opt_frame, "Decimation %:", self.vars["decimation"], 0.0, 1.0, 0)

        # --- 5. Actions Frame ---
        action_frame = ttk.LabelFrame(frame, text="Actions", padding=10)
        action_frame.grid(row=4, column=0, sticky="nsew", pady=15)
        action_frame.columnconfigure(0, weight=1)
        
        self.generate_button = ttk.Button(action_frame, text="GENERATE MESH", command=self.start_generation_thread, style="Accent.TButton")
        self.generate_button.grid(row=0, column=0, sticky="ew", pady=5, ipady=10)
        
        self.save_button = ttk.Button(action_frame, text="Save Final Mesh", state="disabled", command=self.save_final_mesh)
        self.save_button.grid(row=1, column=0, sticky="ew", pady=5)
        
    def create_output_widgets(self):
        """Populates the right-hand output panel."""
        frame = self.output_frame
        frame.rowconfigure(0, weight=5) # Notebook takes up most space
        frame.rowconfigure(1, weight=0) # Stats are small
        frame.rowconfigure(2, weight=1) # Log takes up some space
        frame.columnconfigure(0, weight=1)
        
        # --- 1. Notebook (Tabs) ---
        self.notebook = ttk.Notebook(frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", pady=5)
        
        # Video Tab
        self.video_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.video_tab, text="Video Preview")
        self.video_tab.rowconfigure(0, weight=1)
        self.video_tab.columnconfigure(0, weight=1)
        self.video_tab.columnconfigure(1, weight=1)
        self.video_label = ttk.Label(self.video_tab, text="Select a folder to see video preview.", anchor="center")
        self.video_label.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.play_button = ttk.Button(self.video_tab, text="Play", state="disabled", command=self.play_video)
        self.play_button.grid(row=1, column=0, sticky="n", padx=5, pady=5)
        self.pause_button = ttk.Button(self.video_tab, text="Pause", state="disabled", command=self.pause_video)
        self.pause_button.grid(row=1, column=1, sticky="n", padx=5, pady=5)
        
        # 3D Preview Tab
        self.preview_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.preview_tab, text="3D Preview")
        self.preview_button = ttk.Button(self.preview_tab, text="Show 3D Preview (in new window)", state="disabled", command=self.show_3d_preview)
        self.preview_button.pack(expand=True, anchor="center")

        # --- 2. Stats Frame ---
        stats_frame = ttk.LabelFrame(frame, text="Stats", padding=10)
        stats_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.columnconfigure(1, weight=1)
        stats_frame.columnconfigure(2, weight=1)

        self.frames_label = ttk.Label(stats_frame, text="Frames: N/A")
        self.frames_label.grid(row=0, column=0, sticky="w", padx=5)
        self.vertices_label = ttk.Label(stats_frame, text="Vertices: N/A")
        self.vertices_label.grid(row=0, column=1, sticky="n", padx=5)
        self.faces_label = ttk.Label(stats_frame, text="Faces: N/A")
        self.faces_label.grid(row=0, column=2, sticky="e", padx=5)

        # --- 3. Log Frame ---
        log_frame = ttk.LabelFrame(frame, text="Log", padding=10)
        log_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=5, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # --- Redirect stdout ---
        logger = TextLogger(self.log_text)
        sys.stdout = logger
        sys.stderr = logger
        
        print("Welcome to T1MESCULPTURES!")
        print(f"Persistent cache directory at {app_utils.get_app_cache_dir()}")

    # --- Helper Functions ---
    
    def create_slider(self, parent, text, variable, from_, to, row):
        """Helper to create a label, slider, and value display."""
        
        # 1. Create the Label (no change)
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        
        # 2. Create the Slider (no change)
        slider = ttk.Scale(parent, from_=from_, to=to, variable=variable)
        slider.grid(row=row, column=1, sticky="ew", padx=5)
        
        # 3. Create the Editable Entry Box (replaces Label)
        # We give it a fixed width to keep the layout clean
        value_entry = ttk.Entry(parent, width=7)
        value_entry.grid(row=row, column=2, sticky="e", padx=5)

        # --- Two-Way Binding Logic ---
        
        # Check if we're dealing with an Integer or a Float
        is_int = isinstance(variable, tk.IntVar)

        # 4. Function: Sync the Entry Box (when slider moves)
        def sync_entry_from_slider(slider_value_str):
            # Format the value from the slider
            val = float(slider_value_str)
            if is_int:
                text = f"{int(val)}"  # e.g., "127"
            else:
                text = f"{val:.2f}"   # e.g., "0.50"
            
            # Set the text in the entry box
            value_entry.delete(0, tk.END)
            value_entry.insert(0, text)
            
        # 5. Function: Sync the Slider (when user types in box)
        def sync_variable_from_entry(event):
            try:
                # Get the raw text from the box
                raw_text = value_entry.get()
                
                # Convert it to the correct number type
                if is_int:
                    new_value = int(raw_text)
                else:
                    new_value = float(raw_text)
                
                # Clamp the value to the slider's min/max range
                if new_value < from_: new_value = from_
                if new_value > to: new_value = to
                
                # Set the main variable. This will AUTOMATICALLY move the slider.
                variable.set(new_value)
                
                # Re-format the text in the box (e.g., if user typed "5" -> "5.00")
                sync_entry_from_slider(str(new_value))

            except ValueError:
                # If user types "abc", just reset the box to the variable's last good value
                sync_entry_from_slider(str(variable.get()))
        
        # --- Link everything together ---
        
        # Link slider movement to update the entry box
        slider.configure(command=sync_entry_from_slider)
        
        # Link the "Enter" key and "clicking away" to update the slider
        value_entry.bind("<Return>", sync_variable_from_entry)
        value_entry.bind("<FocusOut>", sync_variable_from_entry)
        
        # Set the entry box's initial value when the app starts
        sync_entry_from_slider(str(variable.get()))

    def browse_folder(self):
        """Open a dialog to choose a folder and load video preview."""
        path = filedialog.askdirectory()
        if not path:
            return
            
        self.vars["path"].set(path)
        print(f"Selected folder: {path}")
        
        # --- Load video preview ---
        try:
            # We load the frames here to use for the video preview
            print("Loading frames for video preview...")
            self.frames_list = data_loader.load_frames(
                path,
                self.vars["filetype"].get(),
                self.vars["threshold"].get()
            )
            if not self.frames_list:
                self.video_label.configure(text="No images found in folder.")
                return
                
            self.frames_label.configure(text=f"Frames: {len(self.frames_list)}")
            self.show_video_frame(0) # Show the first frame
            self.play_button.config(state="normal")
            self.pause_button.config(state="disabled")
            print("Video preview loaded.")
            self.play_video()
            
        except Exception as e:
            print(f"Error loading video preview: {e}")
            messagebox.showerror("Video Error", f"Could not load images: {e}")
        
        self.elapsed_at_pause = 0.0

    # --- Video Player ---
    
    def show_video_frame(self, frame_index):
        """Displays a single frame in the video label."""
        if frame_index >= len(self.frames_list):
            frame_index = 0
            
        self.current_frame_index = frame_index
        
        # Get the OpenCV image (BGR)
        frame_bgr = self.frames_list[frame_index].image
        
        # Convert BGR to RGB
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize to fit the label
        container_w = self.video_label.winfo_width()
        container_h = self.video_label.winfo_height()
        if container_w < 2 or container_h < 2:
            return # Avoids divide-by-zero if widget isn't drawn ye
        img_w, img_h = img_w, img_h = pil_image.size
        scale = min(container_w / img_w, container_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        pil_image = pil_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        # Convert to PhotoImage and display
        self.video_photo = ImageTk.PhotoImage(pil_image)
        self.video_label.config(image=self.video_photo, text="") # Remove "loading" text

    def play_video(self):
        self.is_video_playing = True
        self.play_button.config(state="disabled")
        self.pause_button.config(state="normal")
        
        self.animation_start_time = time.time()
        
        # Start the video loop
        self.video_loop()

    def video_loop(self):
        if not self.is_video_playing:
            return
            
        # 1. Calculate total elapsed time based on system clock
        current_segment_time = time.time() - self.animation_start_time
        total_elapsed_time = self.elapsed_at_pause + current_segment_time
        
        # 2. Get current FPS from the slider
        fps = self.vars["fps"].get()
        if fps == 0: # Avoid divide by zero
            self.after(100, self.video_loop) # Just wait
            return

        # 3. Calculate which frame *should* be visible
        target_frame_float = total_elapsed_time * fps
        target_frame_index = int(target_frame_float) % len(self.frames_list)
        
        # 4. Show that frame (only if it's not the one we're already on)
        if target_frame_index != self.current_frame_index:
            self.show_video_frame(target_frame_index)
        
        # 5. Schedule the next check (e.g., every 10ms)
        self.after(10, self.video_loop)

    def pause_video(self):
        self.is_video_playing = False
        self.play_button.config(state="normal")
        self.pause_button.config(state="disabled")

        # Store the time that just passed
        self.elapsed_at_pause += (time.time() - self.animation_start_time)

    # --- UI Functions ---
            
    def update_smoothing_controls(self, *args):
        """Show or hide the relevant slider for the chosen smoothing method."""
        method = self.vars["smooth_method"].get()
        if method == 'gaussian':
            self.sigma_frame.grid()
            self.iters_frame.grid_remove()
        elif method == 'constrained':
            self.sigma_frame.grid_remove()
            self.iters_frame.grid()
        else: # 'none'
            self.sigma_frame.grid_remove()
            self.iters_frame.grid_remove()

    # --- Main Processing Logic ---
    
    def start_generation_thread(self):
        """
        This is the main "GENERATE MESH" button command.
        It runs the slow process in a background thread.
        """
        print("="*30)
        print("STARTING GENERATION")
        print("="*30)
        
        # Disable button to prevent double-clicks
        self.generate_button.config(state="disabled", text="PROCESSING...")
        
        # Reset output buttons
        self.preview_button.config(state="disabled")
        self.save_button.config(state="disabled")
        
        # Get a snapshot of all parameters
        self.params = {key: var.get() for key, var in self.vars.items()}
        self.params["smooth_kwargs"] = {}
        if self.params["smooth_method"] == 'gaussian':
            self.params["smooth_kwargs"]['sigma'] = self.vars["smooth_sigma"].get()
        elif self.params["smooth_method"] == 'constrained':
            self.params["smooth_kwargs"]['max_iters'] = self.vars["smooth_iters"].get()
            
        # Launch the worker thread
        self.worker_thread = threading.Thread(
            target=self._run_processing_task,
            args=(self.params,) # Pass params to the thread
        )
        self.worker_thread.start()

    def _run_processing_task(self, params):
        """
        THE WORKER THREAD - This runs in the background.
        It must not touch any GUI widgets directly.
        """
        try:
            total_start_time = time.time()
            
            # --- Get persistent cache directory for this project ---
            project_cache_dir = app_utils.get_project_cache_dir(params['path'])
            print(f"Using project cache: {project_cache_dir}")
            
            volume_cache_path = os.path.join(project_cache_dir, "smoothed_volume.npy")
            self.raw_mesh_cache_path = os.path.join(project_cache_dir, "raw_mesh.ply")
            
            # --- STAGE 1 (Fast, no cache) ---
            if not self.frames_list:
                print("Error: No frames loaded.")
                raise ValueError("Frame list is empty. Please select a folder.")
            print("Frames already loaded.")
            
            # --- STAGE 2: VOLUME (Cached) ---
            if os.path.exists(volume_cache_path):
                print("Loading cached volume from disk...")
                smoothed_volume = np.load(volume_cache_path)
            else:
                print("No volume cache found. Running processing...")
                pointcloud, totalFrames, h, w = volume_processor.create_pointcloud_scaffold(
                    self.frames_list, params['scale_factor']
                )
                pointcloud = volume_processor.fill_pointcloud(pointcloud, self.frames_list, h, w)
                rescaled_volume = volume_processor.scale_volume(
                    pointcloud, totalFrames, params['fps'], h, w
                )
                smoothed_volume = volume_processor.smooth_volume(
                    rescaled_volume, params['smooth_method'], **params['smooth_kwargs']
                )
                if smoothed_volume is not None:
                    np.save(volume_cache_path, smoothed_volume)
                    print(f"Saved volume to cache: {volume_cache_path}")
                else:
                    raise ValueError("Volume processing failed.")
                
            vertices = None
            faces = None
            raw_surface = None
            # --- STAGE 3: MESH (Cached) ---
            if os.path.exists(self.raw_mesh_cache_path):
                print("Loading cached raw mesh from disk...")
                try:
                    raw_surface = pv.read(self.raw_mesh_cache_path)
                    vertices = raw_surface.points
                    faces = raw_surface.faces.reshape(-1, 4)[:, 1:]
                    print("Cache loaded successfully.")
                except Exception as e:
                    print(f"Failed to read mesh cache: {e}. Re-generating...")
                    # Ensure variables are None if cache read fails
                    vertices = None
                    faces = None
                    raw_surface = None
           # If cache didn't exist or failed to load, generate the mesh
            if vertices is None: # Check if vertices is still None
                print("No mesh cache found or cache failed. Running mesh extraction...")
                # extract_mesh now returns repaired data if needed
                vertices, faces = mesh_processor.extract_mesh(smoothed_volume)
                
                if vertices is not None and faces is not None:
                    # Save the raw, potentially repaired mesh to cache
                    raw_surface = pv.PolyData(vertices.astype(np.float32), np.hstack((np.full((faces.shape[0], 1), 3), faces)))
                    raw_surface.save(self.raw_mesh_cache_path)
                    print(f"Saved raw mesh to cache.")
                else:
                    # Mesh extraction failed
                    raise ValueError("Mesh extraction failed.")

            if vertices is None or faces is None:
                 raise ValueError("Could not load or generate mesh data.")
                
             # ---  STAGE 4: OPTIMIZATION ---
            print("Running final optimization...")
            # Use the decimation value passed in the 'params' dictionary
            reduction = params['decimation']
            final_surface = mesh_processor.optimize_mesh(
                vertices,
                faces,
                reduction
            )
            if final_surface is None:
                raise ValueError("Mesh optimization failed.")

            print(f"--- Generation complete in {time.time() - total_start_time:.2f}s ---")
            
            # --- Schedule GUI update on the main thread ---
            self.after(0, self._on_generation_complete, final_surface)

        except Exception as e:
            # --- Handle errors and schedule an error message ---
            print(f"--- ERROR IN WORKER THREAD ---")
            print(e)
            self.after(0, self._on_generation_failed, str(e))

    def _on_generation_complete(self, final_mesh_result):
        """
        GUI UPDATE - This runs on the main thread.
        Called by the worker when processing is done.
        """
        print("Updating GUI...")
        self.generate_button.config(state="normal", text="GENERATE MESH")
        self.preview_button.config(state="normal")
        self.save_button.config(state="normal")

        # --- Store the final result ---
        self.final_mesh_result = final_mesh_result
        
        # Update stats based on the FINAL optimized mesh
        self.vertices_label.config(text=f"Vertices: {final_mesh_result.n_points}")
        self.faces_label.config(text=f"Faces: {final_mesh_result.n_cells}")

    def _on_generation_failed(self, error_message):
        """
        Main GUI update if an error occurred.
        """
        print("Generation failed. Resetting GUI.")
        self.generate_button.config(state="normal", text="GENERATE MESH")
        messagebox.showerror("Processing Error", f"An error occurred:\n{error_message}")

    def get_optimized_mesh(self):
        """
        Helper function to retrieve the mesh generated by the background thread.
        """
        if hasattr(self, 'final_mesh_result') and self.final_mesh_result is not None:
            return self.final_mesh_result
        else:
            messagebox.showerror("Error", "No mesh result found. Please run 'GENERATE MESH' first.")
            return None

    def show_3d_preview(self):
        """Run optimization and show the PyVista plotter."""
        print("Running optimization for preview...")
        final_surface = self.get_optimized_mesh()
        
        if final_surface:
            print("Displaying 3D preview in new window...")
            plotter = pv.Plotter()
            plotter.add_mesh(final_surface, show_edges=True)
            plotter.show()
            print("Preview window closed.")

    def save_final_mesh(self):
        """Run optimization and open a 'Save As' dialog."""
        print("Running optimization for saving...")
        final_surface = self.get_optimized_mesh()
        
        if final_surface:
            # Open "Save As" dialog
            default_name = f"{self.vars['output_name'].get()}_{int(self.vars['decimation'].get()*100)}percent.stl"
            filepath = filedialog.asksaveasfilename(
                initialfile=default_name,
                defaultextension=".stl",
                filetypes=[("STL Mesh", "*.stl"), ("All Files", "*.*")]
            )
            
            if filepath:
                try:
                    final_surface.save(filepath)
                    print(f"Final mesh saved to: {filepath}")
                    messagebox.showinfo("Success", f"Mesh saved to {filepath}")
                except Exception as e:
                    print(f"Error saving file: {e}")
                    messagebox.showerror("Save Error", f"Could not save file: {e}")


if __name__ == "__main__":
    app = T1mesculpturesApp()
    
    sv_ttk.set_theme("dark")  # Set the theme

    app.mainloop()