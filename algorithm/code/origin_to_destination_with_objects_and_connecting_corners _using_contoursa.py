import cv2 as cv
import numpy as np
from scipy.spatial import distance
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon


class Frame:
    def __init__(self, filepath):
        self.threshhold = 127
        self.filepath = filepath
        self.image = cv.imread(filepath)
        self.imagegray = self.getGrayImage(self.image)
        self.mask = self.getThreshhold()
        self.outlines = self.getOutlines()
        self.contours = self.getContours()
        self.boundaries = self.getBoundaries()
        self.density = self.getDensity()
        self.corners = self.getCorners()
        self.nextFrame = None
        self.intersectingBoundaries = []
        self.pixelConnections = []
        self.cornerConnections = []

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

    def getBoundaries(self):
        boundRect = [None]*len(self.contours)
        for i in range(len(self.contours)):
             boundRect[i] = cv.boundingRect(self.contours[i])
        return(boundRect)
    
    def getCorners(self):
        #
        #
        # OPENCV ALSO HAS approxPolyDP()
        # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
        # Maybe switch?
        #
        #

        # modify the data type 
        # setting to 32-bit floating point 
        img_gray = np.float32(self.imagegray)

        corners = []
        for cntr in self.contours:
            cntr_2d = cntr.reshape(cntr.shape[0], 2)
            found_corners = approximate_polygon(cntr_2d, tolerance=1.5)
            corners.append(found_corners)
        for crnr in corners:
            for p in crnr:
                 cv.circle(self.image, (p[0],p[1]), 3,255,-1)
        # Showing the result
        # cv.imshow("Polygons", self.image)
        # cv.waitKey(0)
        return corners
    
    def setNextFrame(self, Frame):
        self.nextFrame = Frame
        self.getIntersectingBoundaries(self.nextFrame)
        self.connectContours(self.nextFrame)
        self.connectCorners(self.nextFrame)

    def intersection(self, a,b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w<0 or h<0: return (0,0,0,0)
        return (x, y, w, h)

    def getIntersectingBoundaries(self, ConnectionFrame):
        self.intersectingBoundaries = []
        for i in range(len(self.boundaries)):
            biggestSharedArea = 0
            tempIntersectingBoundaries = None
            for o in range(len(ConnectionFrame.boundaries)):
                intersection = self.intersection(self.boundaries[i], ConnectionFrame.boundaries[o])
                sharedarea = intersection[2] * intersection[3]
                if(biggestSharedArea < sharedarea): 
                    biggestSharedArea = sharedarea
                    tempIntersectingBoundaries = o
            self.intersectingBoundaries.append(tempIntersectingBoundaries)
        # print(self.intersectingBoundaries)

    
    def maskOiriginAndDestinationContours(self, OriginFrame, OriginFrameContourIndex, DestinationFrame,  DestinationFrameContourIndex):
        contouronmask = np.zeros((540,540), np.uint8)
        cv.drawContours(contouronmask,OriginFrame.contours,OriginFrameContourIndex,(255,0,0),1)
        contouronmask = cv.merge((DestinationFrame.mask, np.zeros((540,540), np.uint8), contouronmask))
        # Showing the result
        cv.imshow("Contour on mask", contouronmask)
        cv.waitKey(0)
        nonzeroMask = cv.findNonZero(DestinationFrame.mask)
        nonzeroMask2D = nonzeroMask.reshape(nonzeroMask.shape[0], 2)
        # print(nonzeroMask2D)
        # print(OriginFrame.contours[OriginFrameContourIndex])
        # print(nonzeroMask[0])
        nonzeromaskimg = np.zeros((540,540), np.uint8)
        for pixel in nonzeroMask:
            cv.circle(nonzeromaskimg, pixel[0], 1, 255, -1)
        # Showing the result
        print(DestinationFrame.mask.shape)
        print(nonzeroMask2D.shape)
        print(OriginFrame.contours[OriginFrameContourIndex].shape)
        cv.imshow("Mask", nonzeromaskimg)
        cv.waitKey(0) 
        o_inner = []
        o_outer = []
        print("OriginFrame.contours, length: ", len(OriginFrame.contours[OriginFrameContourIndex]))
        # Annahme: DestinationFrame.mask und OriginFrame.contours[OriginFrameContourIndex] sind NumPy-Arrays

        # Konvertiere die Maske von DestinationFrame in ein 2D-Array
        nonzeroMask2D = np.argwhere(DestinationFrame.mask)

        # Finde die Punkte, die in der Maske liegen und die, die außerhalb liegen
        inMask = np.isin(OriginFrame.contours[OriginFrameContourIndex][:, 0], nonzeroMask2D).all(axis=1)
        o_inner = OriginFrame.contours[OriginFrameContourIndex][inMask]
        o_outer = OriginFrame.contours[OriginFrameContourIndex][~inMask]
        '''
        for contourPoint in OriginFrame.contours[OriginFrameContourIndex]:
            inMask = False
            for maskPoint in nonzeroMask2D:
                if (contourPoint[0] == maskPoint).all():
                    # print(contourPoint[0], "is in mask! Maskpoint:", maskPoint)
                    inMask = True
                    break
                    
            if inMask: o_inner.append(contourPoint)
            else: o_outer.append(contourPoint)
        nonzeroMask = cv.findNonZero(OriginFrame.mask)
        nonzeroMask2D = nonzeroMask.reshape(nonzeroMask.shape[0], 2)
        '''
        d_inner = []
        d_outer = []
        print("DestinationFrame.contours, length: ", len(DestinationFrame.contours[DestinationFrameContourIndex]))
        # Annahme: DestinationFrame.mask und OriginFrame.contours[OriginFrameContourIndex] sind NumPy-Arrays

        # Konvertiere die Maske von DestinationFrame in ein 2D-Array
        nonzeroMask2D = np.argwhere(OriginFrame.mask)

        # Finde die Punkte, die in der Maske liegen und die, die außerhalb liegen
        inMask = np.isin(DestinationFrame.contours[DestinationFrameContourIndex][:, 0], nonzeroMask2D).all(axis=1)
        d_inner = DestinationFrame.contours[DestinationFrameContourIndex][inMask]
        d_outer = DestinationFrame.contours[DestinationFrameContourIndex][~inMask]
        '''
        for contourPoint in DestinationFrame.contours[DestinationFrameContourIndex]:
            inMask = False
            for maskPoint in nonzeroMask2D:
                if (contourPoint[0] == maskPoint).all():
                    # print(contourPoint[0], "is in mask! Maskpoint:", maskPoint)
                    inMask = True
                    break
                    
            if inMask: d_inner.append(contourPoint)
            else: d_outer.append(contourPoint)
        for contourPoint in OriginFrame.contours[OriginFrameContourIndex]:
            # print(contourPoint[0 ])
            if contourPoint[0] in nonzeroMask2D:
                o_inner.append(contourPoint)
            else:
                o_outer.append(contourPoint)
        nonzeroMask = np.transpose(np.nonzero(OriginFrame.mask))
        d_inner = []
        d_outer = []
        for contour in DestinationFrame.contours[DestinationFrameContourIndex]:
            if contour in nonzeroMask:
                d_inner.append(contour)
            else:
                d_outer.append(contour)

        '''
        contourSplit = np.zeros((540,540, 3), np.uint8)
        for pixel in o_inner:
            cv.circle(contourSplit, pixel[0], 1, (255, 0, 0), -1)
        for pixel in o_outer:
            cv.circle(contourSplit, pixel[0], 1, (0, 255, 0), -1)
        for pixel in d_inner:
            cv.circle(contourSplit, pixel[0], 1, (0, 255, 0), -1)
        for pixel in d_outer:
            cv.circle(contourSplit, pixel[0], 1, (255, 0, 0), -1)
        cv.imshow("Split Contours", contourSplit)
        cv.waitKey(0)
        
        '''
        o_outer = cv.subtract(cv.normalize(OriginFrame.contours[OriginFrameContourIndex], OriginFrame.contours[OriginFrameContourIndex], 0.0, 1.0, cv.NORM_MINMAX), cv.normalize(DestinationFrame.mask, DestinationFrame.mask, 0.0, 1.0, cv.NORM_MINMAX))
        o_inner = cv.subtract(cv.normalize(OriginFrame.contours[OriginFrameContourIndex], OriginFrame.contours[OriginFrameContourIndex], 0.0, 1.0, cv.NORM_MINMAX), (1 - cv.normalize(DestinationFrame.mask, DestinationFrame.mask, 0.0, 1.0, cv.NORM_MINMAX)))
        d_outer = cv.subtract(cv.normalize(DestinationFrame.contours[DestinationFrameContourIndex], DestinationFrame.contours[DestinationFrameContourIndex], 0.0, 1.0, cv.NORM_MINMAX), cv.normalize(OriginFrame.mask, OriginFrame.mask, 0.0, 1.0, cv.NORM_MINMAX))
        d_inner = cv.subtract(cv.normalize(DestinationFrame.contours[DestinationFrameContourIndex], DestinationFrame.contours[DestinationFrameContourIndex], 0.0, 1.0, cv.NORM_MINMAX), (1 - cv.normalize(OriginFrame.mask, OriginFrame.mask, 0.0, 1.0, cv.NORM_MINMAX)))
        '''
        return (o_outer, o_inner, d_outer, d_inner)
    
    def findClosestNode(self, node, nodes):
        closest_index = distance.cdist([node], nodes).argmin()
        return closest_index

    def connectContours(self, ConnectionFrame):
        for index, contour in enumerate(self.contours):
            (o_outer, o_inner, d_outer, d_inner) = self.maskOiriginAndDestinationContours(self, index, ConnectionFrame, self.intersectingBoundaries[index])
            print("countour", index, "o_outer len:", len(o_outer), "o_inner len:",len(o_inner))
            print("countour", index, "d_outer len:", len(d_outer), "d_inner len:",len(d_inner))
        
            # Calculate Distances
            # First we calculate from Origin to Destination
            tempPixelConnections = []
            for i in range(0,len(o_outer)):
                # print(o_outer[i])
                x = o_outer[i][0][0]
                y = o_outer[i][0][1]
                closest = self.findClosestNode((x, y), d_inner.reshape(d_inner.shape[0], 2))
                # cv.line(img_ways, (x, y), (d_inner_pix[closest][0], d_inner_pix[closest][1]),255,1)
                tempPixelConnections.append([(x, y), ((d_inner[closest][0][0], d_inner[closest][0][1]))])

            for i in range(0,len(o_inner)):
                x = o_inner[i][0][0]
                y = o_inner[i][0][1]
                closest = self.findClosestNode((x, y), d_outer.reshape(d_outer.shape[0], 2))
                # cv.line(img_ways, (x, y), (d_outer_pix[closest][0], d_outer_pix[closest][1]),255,1)
                tempPixelConnections.append([(x, y), ((d_outer[closest][0][0], d_outer[closest][0][1]))])

            # Now we calculate in reverse order and should a pixel be connected two times, we take the LONGER distance as it must be an important connection else left out

            for i in range(0,len(d_inner)):
                x = d_inner[i][0][0]
                y = d_inner[i][0][1]
                closest = self.findClosestNode((x, y), o_outer.reshape(o_outer.shape[0], 2))
                
                # Find Connected Pixel in our tempPixelConnections Array and compare distances
                position = -1
                for o in range(len(tempPixelConnections)):
                    if tempPixelConnections[o][0] == (o_outer[closest][0][0], o_outer[closest][0][1]):
                        position = o
                        break
                
                if position > -1:
                    # print("Element's Index in the list is:", position)
                    currentDistance = distance.cdist([tempPixelConnections[position][0]], [tempPixelConnections[position][1]])
                    newDistance = distance.cdist([(x, y)], [(o_outer[closest][0][0], o_outer[closest][0][1])])
                    if currentDistance < newDistance:
                        # print("currentDistance < newDistance", currentDistance, "<", newDistance, " is ", currentDistance < newDistance)
                        tempPixelConnections[position][1] = (x, y)
                else:
                    print("ERROR: Connected Pixel is not in the list.")
            

            for i in range(0,len(d_outer)):
                x = d_outer[i][0][0]
                y = d_outer[i][0][1]
                closest = self.findClosestNode((x, y), o_inner.reshape(o_inner.shape[0], 2))
                
                # Find Connected Pixel in our tempPixelConnections Array and compare distances
                position = -1
                for o in range(len(tempPixelConnections)):
                    if tempPixelConnections[o][0] == (o_inner[closest][0][0], o_inner[closest][0][1]):
                        position = o
                        break
                
                if position > -1:
                    # print("Element's Index in the list is:", position)
                    currentDistance = distance.cdist([tempPixelConnections[position][0]], [tempPixelConnections[position][1]])
                    newDistance = distance.cdist([(x, y)], [(o_inner[closest][0][0], o_inner[closest][0][1])])
                    if currentDistance < newDistance:
                        # print("currentDistance < newDistance", currentDistance, "<", newDistance, " is ", currentDistance < newDistance)
                        tempPixelConnections[position][1] = (x, y)
                else:
                    print("ERROR: Connected Pixel is not in the list.")
            
            # Create a black image
            img_ways = np.zeros((540,540,3), np.uint8)
            for connectedPixels in tempPixelConnections:
                cv.line(img_ways, connectedPixels[0], connectedPixels[1],(0,0,255),1) 
                
            self.pixelConnections = tempPixelConnections
            
            # img_ways = cv.normalize(img_ways, img_ways, 0.0, 1.0, cv.NORM_MINMAX)
            
            pix_travel_ways = img_ways
            for pixel in o_outer:
                cv.circle(pix_travel_ways, pixel[0], 1, (255, 0, 0), -1)
            for pixel in d_inner:
                cv.circle(pix_travel_ways, pixel[0], 1, (255, 0, 0), -1)
            for pixel in o_inner:
                cv.circle(pix_travel_ways, pixel[0], 1, (0, 255, 0), -1)
            for pixel in d_outer:
                cv.circle(pix_travel_ways, pixel[0], 1, (0, 255, 0), -1)
            
            for crnr in self.corners:
                for p in crnr:
                    cv.circle(pix_travel_ways, (p[0],p[1]), 3,(255, 255, 255),-1)
            for crnr in ConnectionFrame.corners:
                for p in crnr:
                    cv.circle(pix_travel_ways, (p[0],p[1]), 3,(255, 255, 255),-1)
            
            for i in range(len(self.boundaries)):
                cv.rectangle(pix_travel_ways, (int(self.boundaries[i][0]), int(self.boundaries[i][1])), \
                (int(self.boundaries[i][0]+self.boundaries[i][2]), int(self.boundaries[i][1]+self.boundaries[i][3])), (0, 255, 0), 2)
            for i in range(len(ConnectionFrame.boundaries)):
                cv.rectangle(pix_travel_ways, (int(ConnectionFrame.boundaries[i][0]), int(ConnectionFrame.boundaries[i][1])), \
                (int(ConnectionFrame.boundaries[i][0]+ConnectionFrame.boundaries[i][2]), int(ConnectionFrame.boundaries[i][1]+ConnectionFrame.boundaries[i][3])), (0, 255, 0), 2)

            cv.imshow('Pixel Travel Ways', pix_travel_ways)
            cv.waitKey(0)
    
    def connectCorners(self, ConnectionFrame):
        print(np.asarray(self.pixelConnections).reshape(np.asarray(self.pixelConnections).shape[0], 2))
        for i in range(len(self.corners)):
            contour = self.contours[i].reshape(self.contours[i].shape[0], 2)
            for o in range(len(self.corners[i])):
                previousCorner = o - 1
                if(previousCorner < 0): previousCorner = len(self.corners)
                nextCorner = o + 1
                if(nextCorner >= len(self.corners[i])): nextCorner = 0
                # print(contour)
                # print(self.corners[i][o])
                startContourIndex = np.where((contour == self.corners[i][o]).all(axis=1))
                # print("startCorner = ", self.corners[i][o], " startContour = ", contour[startContourIndex[0][0]], "startContourIndex = ", startContourIndex[0][0])
                previousContourIndex = np.where((contour == self.corners[i][previousCorner]).all(axis=1))
                print("previousCorner = ", self.corners[i][previousCorner], " previousContour = ", contour[previousContourIndex[0][0]], "previousContourIndex = ", previousContourIndex[0][0])
                nextContourIndex = np.where((contour == self.corners[i][nextCorner]).all(axis=1))
                print("nextCorner = ", self.corners[i][nextCorner], " nextContour = ", contour[nextContourIndex[0][0]], "nextCorner = ", nextContourIndex[0][0])
                currentContourIndex = previousContourIndex
                direction = None
                while(currentContourIndex != nextContourIndex):
                    currentContourIndex = currentContourIndex + 1
                    if(currentContourIndex >= len(contour)): currentContourIndex = 0

                
        tempConnectedCorners = []
        # Find Connected Pixel in our tempPixelConnections Array and compare distances
        position = -1
        for o in range(len(self.pixelConnections)):
            if self.pixelConnections[o][0] == (o_outer_pix[closest][0], o_outer_pix[closest][1]):
                position = o
                break
        
        if position > -1:
            self.pixelConnections
        else:
            print("ERROR: Corner doesn't share the position with a connected Pixel.")

origin = Frame('./_export/small/Comp 1_0010.png')
destination = Frame('./_export/small/Comp 1_0025.png')
origin.setNextFrame(destination)