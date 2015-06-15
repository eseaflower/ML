import numpy as np
import glob
import os
from Mammo.MammoData import MammoData

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
    dir = r"E:\work\mammo_data\conv_net_mammo\*.hdr"
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
    
    trainSize = int(len(data)*split)
    return data[0:trainSize], data[trainSize:]


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
    dir = r"E:\work\mammo_data\conv_net_mammo_hires\*100_patch.pkl"
    pklFiles = glob.glob(dir)
    size = 50
    rawData = []
    for pklFile in pklFiles:
        patchSample = loadModel(pklFile)        
        rawData.extend(generateSubPatches(patchSample, size))
        
    rawData = generateFlippedData(rawData)
    data = []
    for sample in rawData:            
        sample.quadratic(size)
        sample.normalize()
#        sample.show()
        data.append(sample)
    
    trainSize = int(len(data)*split)
    return data[0:trainSize], data[trainSize:]


def flatten(data, size):
    X = np.zeros((len(data), size*size), dtype='float32')
    Y = np.zeros((len(data), 2), dtype='float32')
    for i in range(len(data)):
        X[i, :] = data[i].flattenPixeldata()
        Y[i, :] = data[i].getNormalizedCircle()

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
    dir = r"E:\work\mammo_data\conv_net_mammo_hires\*.hdr"
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
