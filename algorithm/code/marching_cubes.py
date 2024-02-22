import cv2 as cv
import numpy as np
import os
from scipy.spatial import distance
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon, marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

class Frame:
    def __init__(self, filepath):
        self.threshhold = 127
        self.filepath = filepath
        self.image = cv.imread(filepath)
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

path = os.fsencode("./t1mesculptures/algorithm/code/_export/small/")
frames = []

with os.scandir(path) as it:
    for entry in it:
        filename = os.fsdecode(entry)
        if filename.endswith(".png") and entry.is_file():
            # print(entry.name, entry.path)
            frames.append(Frame(filename))

print(len(frames))
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
pointcloud = np.zeros((540, 540, len(frames)))

for i, frame in enumerate(frames):
    if i < len(frames) - 1:  # Ensure we don't go out of bounds
        frames[i].setNextFrame(frames[i+1])
        # twoFrameSlice = np.stack((frames[i].mask, frames[i].nextFrame.mask), axis=2)
        pointcloud[:, :, i] = frames[i].mask
        # print(i)

print(pointcloud.shape)

# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = marching_cubes(pointcloud, level=None, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=1, allow_degenerate=False, method='lewiner', mask=None)


'''
# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

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
        output_mesh.vectors[i][j] = verts[f[j], :]

# Save the mesh to file
output_mesh.save('output_mesh.stl')