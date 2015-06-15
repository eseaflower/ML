import os

class Loader(object):

    def __init__(self):
        return

    def Load(self, filename, sep=';'):
        fh = open(filename, 'r', encoding='utf-8')
        allLines = fh.readlines()
        fh.close()        
        resultArray = [[t.strip() for t in l.split(sep)] for l in allLines]
        return resultArray[0], resultArray[1:]