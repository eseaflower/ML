import numpy as np
from Mammo.MammoData import MammoData

class MammoContext(object):
    def __init__(self):
        print("Here!")

    def createModel(self):        
        self.size = 50        
        self.coarseModel = MammoModel(r'RELU100.pkl', self.size)
        self.refineModel = MammoModel(r'refine_RELU100.pkl', self.size)
        print("Create done")

    def predict(self, header, bindata):
        print("Predict")
        mammoSample = MammoData()
        mammoSample.createHeader(header)
        mammoSample.createPixeldata(np.fromstring(bindata, dtype=np.dtype('uint16')))
        return self.internal_predict(mammoSample)        

    
    def internal_predict(self, mammoSample):
        mammoSample.resample(300) # This is hard coded...
        coarseSample = mammoSample.clone()
        cX, cY = self.predict_step(coarseSample, self.coarseModel)
        # cX, cY are normalized coordinates in the coarseSample space.
        # Transform into original data space.
        cInv = coarseSample.transform.clone()
        cInv.invert()
        pX, pY = cInv.apply(cX*coarseSample.width, cY*coarseSample.height)
        
        # Transform into mammoSample space.
        wX, wY = mammoSample.transform.apply(pX, pY)
        mammoSample.crop(wX, wY, self.size, self.size)        
        # Get refined prediction.
        rX, rY = self.predict_step(mammoSample, self.refineModel)
        rInv = mammoSample.transform.clone()
        rInv.invert()
        tX, tY = rInv.apply(rX*mammoSample.width, rY*mammoSample.height)
        return {'x':float(tX) , 'y':float(tY)}

    def predict_step(self, mammoSample, model):
        xStart, yStart, resampleWidth, resampleHeight = mammoSample.quadratic(self.size)
        mammoSample.normalize()
        X = mammoSample.flattenPixeldata()
        X = X.reshape((1, X.shape[0]))
        p = model.predict(X)
        if mammoSample.hasCircle:
            t = mammoSample.getNormalizedCircle()
            diff = t - p[0]
            error = np.sum(diff ** 2)
            print("Predict error: {0}".format(error))


        px = np.round(p[0,0] * mammoSample.width)
        py = np.round(p[0,1] * mammoSample.height)
        pxCopy = np.array(mammoSample.pixelData, copy=True)
        pxCopy[py, px] = 2*np.max(pxCopy)

        #plt.set_cmap('gray')
        #plt.imshow(pxCopy)
        #plt.show()
        return p[0, 0], p[0, 1]
        #return {'x':float((p[0,0]*mammoSample.width - xStart)/resampleWidth) , 'y':float((p[0,1]*mammoSample.height - yStart)/resampleHeight)}

    
class MammoModel(object):
    def __init__(self, modelFilename, size):
        self.size = size
        self.presistenceManager = PersistenceManager()
        self.presistenceManager.set_filename(modelFilename)
        self.x, self.y, self.classifier = self.presistenceManager.load_model()
        # Create the theano predictor function.
        self.predictor = theano.function(inputs=[self.x],
                                         outputs = self.classifier.y_pred)
            
    def predict(self, xPredict):
        return self.predictor(xPredict)
