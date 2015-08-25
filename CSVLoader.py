import os

class Loader(object):

    def __init__(self):
        return

    def Load(self, filename, sep=';'):
        fh = open(filename, 'r', encoding='utf-8')
        allLines = fh.readlines()        
        # First line may contain a byte order marker, remove it.
        if len(allLines) > 0:
            allLines[0] = allLines[0].replace('\ufeff','')
        fh.close()        
        resultArray = [[t.strip() for t in l.split(sep)] for l in allLines]
        return resultArray[0], resultArray[1:]
    
    def LoadAsItems(self, filename, sep=';'):
        headers, data = self.Load(filename, sep)
        result = []
        for row in data:
            item = dict()
            for key, value  in zip(headers,row):
                item[key] = value
            result.append(item)
        return result