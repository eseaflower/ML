import numpy
import time
import theano
import IDXLoader


class Stopwatch(object):
    def __init__(self):
        self.startTime = 0
        self.stopTime = 0
        self.ellapsedTime = 0
        self.running = False
        return
    def Start(self):
        self.startTime = time.time()
        self.running = True
        return        
    
    def Stop(self):
        self.stopTime = time.time()
        self.ellapsedTime = self.stopTime - self.startTime
        return self.ellapsedTime

    def Ellapsed(self):
        if self.running:
            return time.time() - self.startTime
        return self.ellapsedTime
                        
    def Reset(self):
        self.startTime = 0
        self.stopTime = 0
        self.ellapsedTime = 0

class MNISTLoader(object):
    """
    Class for loading MNIST data
    """
    def __init__(self, *args, **kwargs):
        return

    def Load(self, dataFilename, labelFilename, splitFraction=1.0):
        l = IDXLoader.Loader()

        t = Stopwatch()
        t.Start()
        
        train_labels = l.Load(labelFilename)
        train_data = l.Load(dataFilename, theano.config.floatX)
        print("Starting normalize")
        train_data = self.Normalize(train_data, (0., 255), (-1., 1))
        t.Stop()
        print("Done loading data in {0} s".format(t.Ellapsed()))
        
        train_x, test_x = self.Split(train_data, splitFraction)
        train_y, test_y = self.Split(train_labels, splitFraction)
        
        print("Training samples {0}, Test samples {1}".format(train_y.shape[0], test_y.shape[0]))

        return self.MakeShared(train_x), self.MakeShared(train_y, castType='int32'), \
            self.MakeShared(test_x), self.MakeShared(test_y, castType='int32')
     
    def Normalize(self, data, range_in, range_out):        
        """
            data: numpy array of values
            range_in: tuple (min, max) of value ranges in
            range_out: tuple (min, max) of desired value range out            
        """
        in_delta = range_in[1] - range_in[0]
        # Remove input bias
        tmp = numpy.subtract(data, range_in[0])
        out_delta = range_out[1] - range_out[0]
        factor = out_delta / in_delta
        tmp = numpy.multiply(tmp, factor)
        tmp = numpy.add(tmp, range_out[0])
        return tmp

    def Split(self, data, fraction):
        """
        data: array of data
        fraction: Defines the split of the data
        """      
        a_size = data.shape[0] * fraction
        return data[0:a_size], data[a_size:data.shape[0]]

    def MakeShared(self, data, borrow=True, castType=None):        
        if data.dtype != theano.config.floatX:
            data = numpy.array(data, dtype=theano.config.floatX)
        result = theano.shared(data, borrow=borrow)
        if castType != None:
            return theano.tensor.cast(result, castType)
        return result

