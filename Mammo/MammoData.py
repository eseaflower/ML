import numpy as np
import scipy.ndimage as nimg
import matplotlib.pyplot as plt
import json
import pickle
from Mammo.Transform import Transform, Translate, Scale


def unpickleMammoData(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def pickleMammoData(filename, data):
    with open(filename, 'wb') as f:
        return pickle.dump(data, f)


class MammoData(object):
    def __init__(self):
        self.baseFilename = ""
        self.transform = Transform()

    def clone(self):
        result = MammoData()
        result.baseFilename = self.baseFilename
        result.width = self.width
        result.height = self.height
        result.hasCircle = self.hasCircle
        if self.hasCircle:
            result.circleX = self.circleX
            result.circleY = self.circleY
            
        result.transform = self.transform.clone()
        result.pixelData = np.copy(self.pixelData)
        return result
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            
    def load(self, baseFilename):
        self.baseFilename = baseFilename
        headerFile = self.baseFilename + ".hdr"
        pixelFile = self.baseFilename + ".pxl"
        with open(headerFile) as f:             
            jsonHeader = "".join(f.readlines())
            self.createHeader(jsonHeader)
        self.createPixeldata(np.fromfile(pixelFile, dtype=np.dtype('uint16')))        

    def createHeader(self, jsonHeader):
        header = json.loads(jsonHeader)
        self.width = int(header["width"])
        self.height = int(header["height"])
        self.transform = Transform()
        jsonCircle = header.get("circle")        
        self.hasCircle = False
        if jsonCircle:
            circle = json.loads(jsonCircle)
            self.hasCircle = True
            self.circleX = int(circle["x"])
            self.circleY = int(circle["y"])
            

    def createPixeldata(self, nparr):
        self.pixelData = nparr.reshape((self.height, self.width)).astype('float32')

    def resample(self, size):
        zf = np.max([self.width, self.height]) / size
        zf = 1/zf
        self.pixelData = nimg.zoom(self.pixelData, zf)
        newWidth = self.pixelData.shape[1]
        newHeight = self.pixelData.shape[0]
        #if self.hasCircle:
            #xf = newWidth / self.width
            #yf = newHeight / self.height
            

            #self.circleX = int(np.round(self.circleX * xf))
            #self.circleY = int(np.round(self.circleY * yf))
            #self.transform.scale(xf, yf)
        xf = newWidth / self.width
        yf = newHeight / self.height
        self.update_transform(Scale(xf, yf))
        self.width = newWidth
        self.height = newHeight


    def quadratic(self, size):
        self.resample(size)
        resampleWidth = self.width
        resampleHeight = self.height
        quad = np.zeros((size, size), dtype='float32')
        xStart = np.max([(size - self.width) / 2, 0.])
        yStart = np.max([(size -self.height) / 2, 0.])
        quad[yStart:yStart + self.height, xStart:xStart + self.width] = self.pixelData[:, :]
        self.pixelData = quad
        self.width = size
        self.height = size
        self.update_transform(Translate(xStart, yStart))
        #if self.hasCircle:
        #    self.update_transform(Translate(xStart, yStart))
            #self.circleX = self.circleX + xStart
            #self.circleY = self.circleY + yStart
            #self.transform.translate(xStart, yStart)
        
        return xStart, yStart, resampleWidth, resampleHeight
    
    def getAnnotated(self):        
        result = np.array(self.pixelData, copy=True)
        if self.hasCircle:            
            yPos = np.round(self.circleY)
            xPos = np.round(self.circleX)
            if self.is_inside(xPos, yPos):
                result[yPos, xPos] = 2*np.max(result)
        return result

    def getNormalizedCircle(self):
        return np.array([self.circleX / self.width, self.circleY / self.height], dtype='float32')
    
    def flattenPixeldata(self):
        return np.ravel(self.pixelData)        

    def normalize(self):
        self.pixelData -= np.min(self.pixelData)
        self.pixelData /= np.max(self.pixelData)
        
    def clip(self, threshold):
        self.pixelData = np.clip(self.pixelData, threshold, 1.0)
        self.normalize()


    def horizontalFlip(self):
        self.pixelData = np.fliplr(self.pixelData)
        self.update_transform(Scale(-1.0, 1.0))
        self.update_transform(Translate(self.width, 0))        
#        if self.hasCircle:
#            self.update_transform(Scale(-1.0, 1.0))
#            self.update_transform(Translate(self.width, 0))
            #self.circleX = self.width - self.circleX
            #self.transform.mirrorY()            
            #self.transform.translate(self.width, 0)

    def displace(self, x, y):
        xStart = int(np.max([0, x]))
        yStart = int(np.max([0, y]))                
        xEnd = int(self.width + np.min([0, x]))
        yEnd = int(self.height + np.min([0, y]))

        self.pixelData = self.pixelData[yStart:yEnd, xStart:xEnd]
        self.height = self.pixelData.shape[0]
        self.width = self.pixelData.shape[1]
        self.update_transform(Translate(-xStart, -yStart))
        if self.hasCircle:
            #self.update_transform(Translate(-xStart, -yStart))
            #self.circleX -= xStart
            #self.circleY -= yStart
            #self.transform.translate(-xStart, -yStart)
            # Check that we don't cut out the circle marking.
            if self.circleX < 0 or  self.circleY < 0 or self.circleX >= self.width or self.circleY >= self.height:
                self.hasCircle = False
                return False
        
        return True

    def crop(self, centerX, centerY, wantedWidth, wantedHeight):
        xStart = int(centerX - wantedWidth/2.0)
        yStart = int(centerY - wantedHeight/2.0)
        xEnd = xStart + wantedWidth
        yEnd = yStart + wantedHeight

        xStart = np.max([0, xStart])
        yStart = np.max([0, yStart])
        xEnd = np.min([self.width, xEnd])
        yEnd = np.min([self.height, yEnd])

        self.pixelData = self.pixelData[yStart:yEnd, xStart:xEnd]
        self.height = self.pixelData.shape[0]
        self.width = self.pixelData.shape[1]
        self.update_transform(Translate(-xStart, -yStart))
        if self.hasCircle:
            #self.update_transform(Translate(-xStart, -yStart))
            #self.circleX -= xStart
            #self.circleY -= yStart
            #self.transform.translate(-xStart, -yStart)
            # Check that we don't cut out the circle marking.
            if self.circleX < 0 or  self.circleY < 0 or self.circleX >= self.width or self.circleY >= self.height:
                self.hasCircle = False
                return False
        
        return True

    # Make sure the transform is up-to-date
    def update_transform(self, prepend):
        if self.hasCircle:
            self.circleX, self.circleY = prepend.apply(self.circleX, self.circleY)
        self.transform.prepend(prepend)


    def show(self):
        plt.set_cmap('gray')
        plt.imshow(self.getAnnotated())
        plt.show()

    def is_inside(self, x, y):        
        # Check bounds
        return x >= 0 and  y >= 0 and x < self.width and y < self.height