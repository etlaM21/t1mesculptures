import cv2 as cv
import numpy as np
import os
from scipy.spatial import distance
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon, marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import MarchingNumPy.MarchingNumPy as MarchingNumPy
from stl import mesh

THRESHHOLD = 127
FPS = 25    

class Frame:
    def __init__(self, filepath):
        self.threshhold = THRESHHOLD
        self.filepath = filepath
        self.image = cv.imread(filepath)
        self.height, self.width, self.channels = self.image.shape
        self.imagegray = self.getGrayImage(self.image)
        self.mask = self.getThreshhold()
        self.outlines = self.getOutlines()
        self.contours = self.getContours()
        self.nextFrame = None

    def getGrayImage(self, image):
        # Convert to graycsale
        img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return img_gray

    def getThreshhold(self):
        gaussian = cv.GaussianBlur(self.imagegray, (3, 3), 0)
        threshhold, img_treshhold = cv.threshold(self.imagegray, self.threshhold, 255, cv.THRESH_BINARY)
        return img_treshhold

    def getOutlines(self):
        # Blur the image for better edge detection
        img_blur = cv.GaussianBlur(self.imagegray, (3,3), 0)
        img_edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
        img_edges = cv.threshold(img_edges, self.threshhold, 255, cv.THRESH_BINARY)[1]
        return img_edges

    def getDensity(self):
        # Use starting image
        img_altitude = cv.distanceTransform(self.imagegray, cv.DIST_L2, 3)
        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        img_altitude = cv.normalize(img_altitude, img_altitude, 0.0, 1.0, cv.NORM_MINMAX)
        return img_altitude
    
    def getContours(self):
        ret, thresh = cv.threshold(self.imagegray, self.threshhold, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        return contours
    
    def setNextFrame(self, Frame):
        self.nextFrame = Frame
            
# origin = Frame('./algorithm/code/_export/small/Comp 1_0010.png')
# destination = Frame('./algorithm/code/_export/small/Comp 1_0025.png')
# origin.setNextFrame(destination)

# path = os.fsencode("./_export/small")
path = os.fsencode(input("Path to image sequence: "))
FPS = int(input("FPS: "))
outputname = input("Outputname: ") + ".stl"
frames = []

with os.scandir(path) as it:
    for entry in it:
        filename = os.fsdecode(entry)
        if filename.endswith(".png") and entry.is_file():
            # print(entry.name, entry.path)
            frames.append(Frame(filename))

totalFrames = len(frames)

print(f"Total length: {totalFrames} Frames at {FPS} FPS")

'''
pointcloud = np.zeros((540, 540 ,2))

for i, frame in enumerate(frames):
    if i < len(frames) - 1:
        frames[i].setNextFrame(frames[i+1])
        twoFrameSlice = np.stack((frames[i].mask, frames[i].nextFrame.mask))
        pointcloud = np.concatenate((pointcloud,twoFrameSlice.reshape((540, 540, 2))), axis=2)
        print(i)

print(pointcloud.shape)
'''
# Preallocate pointcloud array with the correct dimensions
pointcloud = np.zeros((frames[0].width, frames[0].height, len(frames)))

for i, frame in enumerate(frames):
    if i < len(frames) - 1:  # Ensure we don't go out of bounds
        frames[i].setNextFrame(frames[i+1])
        # twoFrameSlice = np.stack((frames[i].mask, frames[i].nextFrame.mask), axis=2)
        pointcloud[:, :, i] = frames[i].mask

print(f"Pointcloud generated with shape {pointcloud.shape}: min value = {pointcloud.min()}, max value = {pointcloud.max()}")
level = THRESHHOLD
print(f"Processing pointcloud at level {level}")

# Use marching cubes to obtain the surface mesh of these ellipsoids
# vertices (NDArray), geometry (NDArray) =  MarchingNumPy.MarchingCubesLorensen.marching_cubes_lorensen(volume, level=0.0, *, interpolation='LINEAR', step_size=1, resolve_ambiguous=True)
# vertices, faces = MarchingNumPy.marching_cubes_lorensen(np.load("MarchingNumPy/example_data/test_volume.npy"), level=level, interpolation='LINEAR', step_size=1, resolve_ambiguous=True)
vertices, faces = MarchingNumPy.marching_cubes_lorensen(np.pad(pointcloud, 1), level=level, interpolation='COSINE', step_size=0.5, resolve_ambiguous=True)
print(f"Marching Numpy: {len(vertices)} Vertices, {len(faces)} Faces")
maxHeight = (totalFrames / FPS) * (max(np.shape(pointcloud)[0], np.shape(pointcloud)[1]) / FPS)
print(f"Rescaling Output with a maximum height of {maxHeight}")
for i, f in enumerate(vertices): 
    relHeight = vertices[i][2] / totalFrames
    newHeight = relHeight * maxHeight
    vertices[i][2] = newHeight

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes docstring).
'''
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of faces
meshPoly3D = Poly3DCollection([geometry])
meshPoly3D.set_edgecolor('k')
ax.add_collection3d(meshPoly3D)

ax.set_xlim(-10, 550)  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(-10, 550)  # b = 10
ax.set_zlim(0, 250)  # c = 16

plt.tight_layout()
plt.show()
'''

# Create the mesh object
output_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        output_mesh.vectors[i][j] = vertices[f[j], :]

# Save the mesh to file
output_mesh.save(outputname)