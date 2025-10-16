# Current Implementation

## [algorithm/code/v2/time_to_space_pymcubes_pymeshfixy.py](./code/v2/time_to_space_pymcubes_pymeshfixy.py)

## Packages Used

- `cv2` OpenCV for image manipulation
- `numpy` NumPy for list and vector manipulation
- `scipy.ndimage` SciPy for multidimensional image processing
- `pyvista` PyVista for mesh simplification and viewing
- `pymeshfix` PyMeshFix to check if mesh is manifold
- `mcubes` PyMCubes for the MArching Cube algorithm

## 1. Image Processing

1. Find all `.png` images in given path
2. Append each image as `FRAME` object in `frames` list
    1. Save image data (path, height, channels, original image) in Object
    2. Save grayscale version of image in Object
    3. Save mask of image, binary version with global threshhold, in Object

## 2. Data Pre-Paration

1. Downsample `frames` scaling along `[X,Y]` by `scale_factor = 0...1`
    - Scales width and height of list
2. Create initial, empty pointcloud (list) from `[totalFrames, frames.height, frames.width]`

## 3. Point Cloud Generation

1. Fill pointcloud with each frame's mask
   1. X, Y dimension lists are either 0 or 1 depending on binary mask
   2. Z position is determined by frame number

## 4. Z-Axis Scaling

1. `z_zoom_factor` = `(totalFrames / FPS) * max(scaled_width, scaled_height) / totalFrames`, meaning the pointcloud is stretched so that one second of frames is as high as the longes side. Every 45deg line is equal 1 second of movement.
2. Zomming / Stretching is done by `scipy.ndimage.zoom()` which interpolates points in bewtween to make sure the pointcloud always has an equal number of points in each direction

## 5.Pointcloud Smoothing

1. `mcubes.smooth()` is used on the pointcloud to smooth out harsh "edges" before marching cubes

## 6. Mesh Extraction with Marching Cubes

1. `mc.marching_cubes` extracts a mesh from the pointcloud using a marching cubes implementation

## 7. Final Optimization via Mesh Decimation

1. PyVista's `decimate_pro()` is used to decimate the mesh to a target reduction percent

## 8. Last Manifold Check & Reparation

1. PyVista checks if mesh `is_manifold`
2. IF NOT: PyMeshFix `repair()` is used to (hopefully) make Mesh manifold again

## 9. Mesh is saved

1. Mesh is saved to provided path under provided name