## Takes input Images
## Returns list of Frame Objects with threshholded masks

import cv2 as cv
import os

# --- Frame Class (for image processing) ---
# This class handles loading and processing each individual image file.
class Frame:
    def __init__(self, filepath, threshhold):
        self.threshhold = threshhold
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
    
def load_frames(path, filetype, threshhold):
    # --- Image Loading and Processing ---
    frames = []
    try:
        with os.scandir(path) as it:
            sorted_entries = sorted(it, key=lambda entry: entry.name)
            for entry in sorted_entries:
                filename = os.fsdecode(entry.path)
                if filename.endswith(f".{filetype}") and entry.is_file():
                    frames.append(Frame(filename, threshhold))
    except FileNotFoundError:
        print(f"Error: The directory '{path}' was not found.")
        exit()

    if not frames:
        print("Error: No .png files were found in the specified directory.")
        exit()

    totalFrames = len(frames)
    print(f"\nSuccessfully loaded {totalFrames} frames.")
    return frames