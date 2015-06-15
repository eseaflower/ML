import numpy as np 
import Levenshtein
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


