import numpy as np 
#import Levenshtein
import re


def buildLookup(s):
    index = 0
    result = dict()
    for item in s:
        if not item in result:
            result[item] = index
            index += 1
    return result

def buildLookup2(s, indexFunction):
    index = 0
    result = dict()
    for row in s:
        item = indexFunction(row)
        if not item in result:
            result[item] = index
            index += 1
    return result


def buildCodepartLookup2(s,indexFunction, func):
    splitSet = [func(indexFunction(m)) for m in s]
    return buildLookup(splitSet)


def buildCodepartLookup(s, func):
    splitSet = [func(m) for m in s]
    return buildLookup(splitSet)

def splitUpper(item):
    regexp = ' |\^|/|,|\.|\+|_'        
    return [u.upper() for u in re.split(regexp,  item)]

def tfidf(data, index):
    wordDict = dict()
    totalDocuments = 0
    for row in data:
        count = int(row[0])
        totalDocuments += count
        doc = row[index]
        docSplit = splitUpper(doc)        
        seenWords = set()
        for word in docSplit:
            wordEntry = wordDict.get(word)
            if not wordEntry:
                wordEntry = {"wc":0, "dc":0, "tfidf":0.}
                wordDict[word] = wordEntry
            
            wordEntry["wc"] += count            
            if word not in seenWords:
                seenWords.add(word)
                wordEntry["dc"] += count

    
    
    for kv in wordDict.items():
        word = kv[0]
        entry = kv[1]
        score = 0
        if entry["dc"] > 2:
            tf = (entry["wc"] / float(entry["dc"]))    
            idf = np.log(totalDocuments / entry["dc"])    
            score = tf * idf  
        entry["tfidf"] = score
                
    return wordDict


def selectTerms(data, index):
    words = tfidf(data, index)
    sortedWords = sorted(words.items(), key=lambda x:x[1]["dc"], reverse = True)
    terms = []    

    for e in sortedWords:
        if e[1]["dc"] > 500:
            terms.append(e[0])            
    return terms

def computeConditionals(data, index, labels):
        
    labelDict = dict()
    wordDict = dict()

    setSize = 0

    for rowIndex in range(len(data)):
        row = data[rowIndex]
        count = int(row[0])
        label = labels[rowIndex].upper()
        
        setSize += count

        labelEntry = labelDict.get(label)
        if labelEntry == None:
            labelEntry = {"classCount": 0}
            labelDict[label] = labelEntry
        labelEntry["classCount"] += count

        doc = row[index]
        docSplit = splitUpper(doc)        
        seenWords = set()
        
        
        for word in docSplit:
            if word not in seenWords:
                seenWords.add(word)
                wordEntry = wordDict.get(word)
                if not wordEntry:
                    wordEntry = {"count":0, "classDict":dict()}                
                    wordDict[word] = wordEntry
                wordEntry["count"] += count
                classEntry = wordEntry["classDict"].get(label)
                if not classEntry:
                    classEntry = {"count":0}
                    wordEntry["classDict"][label] = classEntry
                classEntry["count"] += count    
    


    totalEntropy = 0.0                
    for k, v in labelDict.items():
        pLabel = v["classCount"] / setSize
        totalEntropy += -pLabel*np.log2(pLabel)            

    
    # Compute conditional probabillties    
    tstDict = dict()
    for k, v in wordDict.items():
        #Compute the probabillity of word k in the dataset.
        totalWordCount = v["count"]
        
        pWord =  totalWordCount/ setSize
        #Compute entropy of the set of examples where k is present. H(C|x=k) = SUM(classes)=>P(Ci|x=k)*log(P(Ci|x=k)
        acc = 0.0
        for cl, clc in v["classDict"].items():
            pc = clc["count"] / totalWordCount            
            acc += -pc * np.log2(pc)
        

        # Estimate entropy when ferature is not present with the global entropy
        fakeIg = totalEntropy - acc*pWord - (1-pWord)*totalEntropy
        tstDict[k] = fakeIg
                        
        
    tstDict = sorted(tstDict.items(), key=lambda x:x[1], reverse=True)
    #print(len(tstDict))
    #input("ldsakjgf")
    numTerms = min(len(tstDict), 300)

    termList = [x[0] for x in tstDict[:numTerms]]
    return termList


def selectTopWords(items, indexFunction, numberOfWords):
    wordCounts = dict()
    for row in items:
        value = indexFunction(row)
        words = splitUpper(value)
        for w in words:
            if w not in wordCounts:
                wordCounts[w] = 1
            else:
                wordCounts[w] = wordCounts[w] + 1

    sortedDictionary = sorted(wordCounts.items(), key=lambda x: x[1], reverse=True)
    toSelect = min(len(sortedDictionary), numberOfWords)
    return [x[0] for x in sortedDictionary[:toSelect]]


class DictionaryFeatureMap2(object):
    def __init__(self, items, indexFunction):
        self.indexFunction = indexFunction
        self.dictionary = buildLookup(items)
        self.dimension = len(self.dictionary)
        return
        
    def getDimension(self):
        return self.dimension
    
    def map(self, row):
        sample = self.indexFunction(row)
        values = splitUpper(sample)
        result = np.zeros(self.dimension, dtype='float32')
        for value in values:
            if value in self.dictionary:
                index = self.dictionary[value]
                result[index] = 1.0
        return result

class CodePartFeatureMap2(object):
    def __init__(self, items, indexFunction, splitFunction):
        self.indexFunction = indexFunction
        self.splitFunction = splitFunction
        self.dictionary = buildCodepartLookup2(items, self.indexFunction, self.splitFunction)
        self.dimension = len(self.dictionary)
        return

    def getDimension(self):
        return self.dimension
    
    def map(self, sample):
        values = splitUpper(self.indexFunction(sample))
        result = np.zeros(self.dimension, dtype='float32')
        for rawValue in values:
            value = self.splitFunction(rawValue)
            if value in self.dictionary:
                index = self.dictionary[value]
                result[index] = 1.0
        return result

class DictionaryLabelMap2(object):
    def __init__(self, items, indexFunction):
        self.indexFunction = indexFunction
        self.dictionary = buildLookup([indexFunction(s).upper() for s in items])      
        self.dimension = 1
        self.range =len(self.dictionary)

    def getDimension(self):
        return self.dimension

    def getRange(self):
        return self.range

    def map(self, value):
        result = np.zeros(self.dimension, dtype='float32')
        result = self.dictionary[self.indexFunction(value).upper()]
        return result

    def inverseMap(self, index):
        for key, value in self.dictionary.items():
            if index == value:
                return key
        raise NameError()





class DictionaryFeatureMap(object):
    def __init__(self, items, index):
        self.dictionary = buildLookup(items)
        self.dimension = len(self.dictionary)
        self.index = index
        return
        
    def getDimension(self):
        return self.dimension
    
    def map(self, sample):
        values = splitUpper(sample[self.index])
        result = np.zeros(self.dimension, dtype='float32')
        for value in values:
            if value in self.dictionary:
                index = self.dictionary[value]
                result[index] = 1.0
        return result

class CodePartFeatureMap(object):
    def __init__(self, items, index, splitFunction):
        self.splitFunction = splitFunction
        self.dictionary = buildCodepartLookup(items, self.splitFunction)
        self.dimension = len(self.dictionary)
        self.index = index        
        return

    def getDimension(self):
        return self.dimension
    
    def map(self, sample):
        values = splitUpper(sample[self.index])
        result = np.zeros(self.dimension, dtype='float32')
        for rawValue in values:
            value = self.splitFunction(rawValue)
            if value in self.dictionary:
                index = self.dictionary[value]
                result[index] = 1.0
        return result

class CodeFeatureMap(object):
    def __init__(self, index):
        self.dimension = 5
        self.index = index
        return

    def getDimension(self):
        return self.dimension
    
    def map(self, sample):
        values = splitUpper(sample[self.index])
        valueOfInteres = values[0]
        result = np.zeros(self.dimension, dtype='float32')
        index = 0
        for part in valueOfInteres:                            
            if part.isdigit():                
                result[index] = int(part) / 10.0            
            index += 1
            if index >= self.dimension:
                break        
        return result

class TermDictionaryFeatureMap(object):
    def __init__(self, termSet, index):
        self.terms = termSet        
        self.dimension = len(self.terms)
        self.index = index
        return
    
    def getDimension(self):
        return self.dimension

    def termDistance(self, term):
        return np.array([Levenshtein.jaro(term, k) for k in self.terms])
    
    def map(self, sample):
        values = splitUpper(sample[self.index])
        result = np.zeros(self.dimension, dtype='float32')
        for value in values:
            result = np.max([result, self.termDistance(value)], axis = 0)
        
        #result[result < 0.99] = 0.0
        return result

class DictionaryLabelMap(object):
    def __init__(self, labels):
        self.dictionary = buildLookup([u.upper() for u in labels])
        self.dimension = 1
        self.range =len(self.dictionary)

    def getDimension(self):
        return self.dimension

    def getRange(self):
        return self.range

    def map(self, value):
        result = np.zeros(self.dimension, dtype='float32')
        result = self.dictionary[value.upper()]
        return result

    def inverseMap(self, index):
        for key, value in self.dictionary.items():
            if index == value:
                return key
        raise NameError()


class FeatureMapBase(object):
    def __init__(self, accessorFunc, valueFunc):
        self._dimension = 0
        self._range = 0        
        if not accessorFunc:
            accessorFunc = lambda x: x
        if not valueFunc:
            valueFunc = lambda x: x
        self.accessorFunc = accessorFunc
        self.valueFunc = valueFunc
    
    def getValues(self, item):
        return self.valueFunc(self.accessorFunc(item))

    @property
    def dimension(self):
        return self._dimension
    
    @property
    def range(self):
        return self._range

    def build(self, dataSet):
        raise NotImplementedError()

    def map(self, item):
        raise NotImplementedError()


class BagOfItemsMap(FeatureMapBase):
    def __init__(self, accessorFunc, valueFunc):
        super().__init__(accessorFunc, valueFunc)
        self.dictionary = dict()
                
    def buildIndexDictionary(self, dataSet):
        # Build a unique set of items.
        uniqueSet = set()
        for sample in dataSet:            
            values = self.getValues(sample)
            for value in values:
                uniqueSet.add(value)
        # With the set of unique items we can build a dictionary
        self.dictionary = dict()
        index = 0
        for value in uniqueSet:
            self.dictionary[value] = index
            index += 1

    def build(self, dataSet):
        self.buildIndexDictionary(dataSet)
        size = len(self.dictionary)
        self._dimension = size
        self._range = size

    def getIndexes(self, item):
        values = self.getValues(item)
        uniqueIndexes = set()
        for value in values:
            index = self.dictionary.get(value)
            if not (index is None):
                uniqueIndexes.add(index)
        result = [v for v in uniqueIndexes]
        return result

    def map(self, item):
        result = np.zeros(self.dimension, dtype='float32')
        indexes = self.getIndexes(item)
        result[indexes] = 1.0
        return result

class NumberMap(FeatureMapBase):
    def __init__(self, dimension, accessorFunc, valueFunc):
        super().__init__(accessorFunc, valueFunc)
        self._dimension = dimension
        self._range = None
    
    def build(self, dataSet):
        pass

    def map(self, item):
        values = self.getValues(item)
        result = np.asarray(values, dtype='float32')
        return result

class LabelMap(BagOfItemsMap):
    def __init__(self, accessorFunc, valueFunc):
        super().__init__(accessorFunc, valueFunc)
        self.inverseDictionary = dict()
    
    def build(self, dataSet):
        super().build(dataSet)
        self._dimension = 1
        for key, value in self.dictionary.items():
            self.inverseDictionary[value] = key

    def map(self, item):
        result = np.zeros(self.dimension, dtype='float32')
        # Use the item index as value
        result[:] = self.getIndexes(item)

    def inverseMap(self, index):
        result = self.inverseDictionary.get(index)
        if result is None:
            raise ValueError()
        return result

class ItemMapper(object):
    def __item__(self, features, label):
        self.featureMappers = featureMappers
        self.labelMapper = label
        self._dimension = sum([m.dimension for m in self.featureMappers])
        self._range = self.labelMapper.range
        
    @property
    def dimension(self):
        return self._dimension
    
    @property
    def range(self):
        return self._range

    def mapFeatures(self, dataSet):
        numberOfSamples = len(dataSet)
        result = np.zeros((numberOfSamples, self.dimension), dtype = 'float32')
        for i in range(numberOfSamples):
            if not i % 100:
                print("Sample {0}".format(i))
            item = dataSet[i]
            itemFeatures = [m.map(item) for m in self.featureMappers]
            result[i] = np.concatenate(itemFeatures)
        return result

    def mapLabels(self, dataSet):
        numberOfSamples = len(dataSet)
        result = np.zeros((numberOfSamples, self._range), dtype = 'float32')
        for i in range(numberOfSamples):
            item = dataSet[i]
            result[i] = self.labelMapper.map(item)
        return result

    def map(self, dataSet):
        return self.mapFeatures(dataSet), self.mapLabels(dataSet)

class mapper(object):
    def __init__(self, features, label):
        self.featureMappers = features
        self.labelMapper = label
        self.dimension = sum([m.getDimension() for m in self.featureMappers])
        self.range = label.getRange()
        return

    def getDimension(self):
        return self.dimension

    def mapX(self, x):
        nSamples = len(x)
        mappedX = np.zeros((nSamples, self.dimension), dtype='float32')
        
        nRawFeatures = len(self.featureMappers)
        for i in range(nSamples):
            if not i % 100:
                print("Sample {0}".format(i))
            
            sample = x[i]
            allFeatures = []
            for fMap in self.featureMappers:                
                allFeatures.append(fMap.map(sample))
            mappedX[i] = np.concatenate(allFeatures)
        
        return mappedX
    
    def mapY(self, y):
        nSamples = len(y)
        mappedY = np.zeros(nSamples, dtype='float32')                
        nRawFeatures = len(self.featureMappers)
        for i in range(nSamples):
            mappedY[i] = self.labelMapper.map(y[i])

        return mappedY

    def map(self, x, y):
        return self.mapX(x), self.mapY(y)
    
    def map2(self, data):
        return self.mapX(data), self.mapY(data)


