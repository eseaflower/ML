import numpy as np
import glob
import os
from Mammo.MammoData import MammoData, unpickleMammoData, pickleMammoData


DataDirectory = r'..\Data\Mammo\mammo_data'


def splitData(data, split=0.9):
    trainSize = int(len(data)*split)
    return data[0:trainSize], data[trainSize:]


def generateTranslated(sample):
    result = [sample]
    axisDisplacement = [-5, 0, 5]
    for y in axisDisplacement:
        for x in axisDisplacement:
            if y==0 and x==0:
                continue
            newSample = sample.clone()
            if newSample.displace(x, y):
                result.append(newSample)

    return result        

def generateFlippedData(data):
    result = []
    for sample in data:
        morph = sample.clone()
        morph.horizontalFlip()
        result.extend([sample, morph])
    return result

def morphTrainingData(data):
    flippedData = generateFlippedData(data)
    result = []
    for sample in flippedData:
        result.extend(generateTranslated(sample))
    return result

def loadData(size, split=0.9):
    #dir = r"E:\work\mammo_data\conv_net_mammo\*.hdr"
    dir = "{0}\conv_net_mammo\*.hdr".format(DataDirectory)
    
    headerFiles = glob.glob(dir)
    rawData = []    
    for headerFile in headerFiles:
        baseFilename = os.path.splitext(headerFile)[0]
        sample = MammoData()        
        sample.load(baseFilename)
        rawData.append(sample)
    
    rawData = morphTrainingData(rawData)
        
    data = []
    for sample in rawData:            
        sample.quadratic(size)
        sample.normalize()
        data.append(sample)
    
    return splitData(data, split)


def generateSubPatches(sample, size):
    
    result = []
    npatches = 10
    rng = np.random.RandomState(1234)
    displace = rng.normal(0.0, 10, (10, 2))

    for d in displace:
        newSample = sample.clone()
        if newSample.crop(newSample.circleX + d[0], newSample.circleY + d[1], size, size):
            result.append(newSample)
    return result

def loadPatchData(size, split=0.9):
    #dir = r"E:\work\mammo_data\conv_net_mammo_hires\*100_patch.pkl"
    dir = "{0}\conv_net_mammo_hires\*100_patch.pkl".format(DataDirectory)
    pklFiles = glob.glob(dir)
    size = 50
    rawData = []
    for pklFile in pklFiles:
        patchSample = unpickleMammoData(pklFile)        
        rawData.extend(generateSubPatches(patchSample, size))
        
    rawData = generateFlippedData(rawData)
    data = []
    for sample in rawData:            
        sample.quadratic(size)
        sample.normalize()
#        sample.show()
        data.append(sample)
           
    return splitData(data, split)


def flatten(data, size):
    X = np.zeros((len(data), size*size), dtype='float32')
    Y = np.zeros((len(data), 2), dtype='float32')
    for i in range(len(data)):
        X[i, :] = data[i].flattenPixeldata()
        Y[i, :] = data[i].getNormalizedCircle()

    return X, Y

def flattenClassification(data, size):
    X = np.zeros((len(data), size*size), dtype='float32')
    Y = np.zeros((len(data),), dtype='float32')
    for i in range(len(data)):
        X[i, :] = data[i].flattenPixeldata()
        val = 0
        if data[i].hasCircle:
            val = 1
        Y[i] = val

    return X, Y


def makePatch(sample):
    patchSize = 100
    # Get a 100x100 patch centered on the circle
    if sample.hasCircle:
        if sample.crop(sample.circleX, sample.circleY, patchSize, patchSize):            
            # The patch still contains the circle
            patch_filename = "{0}_100_patch.pkl".format(sample.baseFilename)
            sample.save(patch_filename)

def handleHiResData():
    #dir = r"E:\work\mammo_data\conv_net_mammo_hires\*.hdr"
    dir = "{0}\conv_net_mammo_hires\*.hdr".format(DataDirectory)
    headerFiles = glob.glob(dir)
    cnt = 1
    for headerFile in headerFiles:
        baseFilename = os.path.splitext(headerFile)[0]
        sample = MammoData()        
        sample.load(baseFilename)
        sample.resample(300)        
        makePatch(sample)
        print("Patch {0} done".format(cnt))
        cnt += 1



def makeClassificationPatch(sample, patchSize, nPatches):

    rng = np.random.RandomState(1234)

    result = []
    # Positive patches
    displace = rng.uniform(-patchSize/2, patchSize/2, size = (nPatches, nPatches))    
    for d in displace:
        newSample = sample.clone()
        if newSample.crop(newSample.circleX + d[0], newSample.circleY + d[1], patchSize, patchSize):
            if (newSample.width == patchSize) and (newSample.height == patchSize):
                result.append(newSample)
    
            
    # Negative patches.
    displaceX = rng.uniform(0, sample.width, size = (nPatches))
    displaceY = rng.uniform(0, sample.height, size = (nPatches))    
    for dX, dY in zip(displaceX, displaceY):
        newSample = sample.clone()
        if not newSample.crop(dX, dY, patchSize, patchSize):
            if (newSample.width == patchSize) and (newSample.height == patchSize):
                result.append(newSample)
                            
    return result


def makeClassificationPatches(data, patchSize):
    nPatches = 10
    result = []
    for sample in data:
        result.extend(makeClassificationPatch(sample, patchSize, nPatches))
    return result

            
def loadClassificationData(split=0.9):
    dir = r"{0}\conv_net_mammo_hires\\".format(DataDirectory)
    resampledDataFilename = "{0}resampled_100.pkl".format(dir)
    
    # Load the resampled data
    data = unpickleMammoData(resampledDataFilename)
    
    # Get mirrored versions of the data
    data = generateFlippedData(data)

    return splitData(data, split)

            
def makeClassificationData():
    #dir = "{0}\conv_net_mammo_hires\*.hdr".format(DataDirectory)
    dir = r"{0}\conv_net_mammo_hires\\".format(DataDirectory)
    pattern = r"{0}*.hdr".format(dir)
    headerFiles = glob.glob(pattern)
    size = 100
    cnt = 1
    data = []
    for headerFile in headerFiles:
        baseFilename = os.path.splitext(headerFile)[0]
        sample = MammoData()        
        sample.load(baseFilename)
        # Normalize data.
        sample.quadratic(100)
        sample.normalize()
        data.append(sample)
        print("Patch {0} done".format(cnt))
        cnt += 1
    
    resampledDataFilename = "{0}resampled_100.pkl".format(dir)
    pickleMammoData(resampledDataFilename, data)



