import random
import featuremapping as fm
import numpy as np
import theano
import theano.tensor as T
import nnlayer
import Trainer



class Apa(object):
    def __init__(self):
        pass
    def printTest(self, value):
        print(value)





class AnatomyData(object):
    def __init__(self, data):
        self.Id = data.get("Id")
        self.Counts = data.get("Counts")
        self.Modality = data.get("Modality")
        self.Code = data.get("Code")
        self.Bodypart = data.get("BodyPart")
        self.Description = data.get("Description")
        self.Label = data.get("Label")
        self.Predicted = data.get("Predicted")
        self.Correct = data.get("Correct")

        

class AnatomyWhitener(object):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    
    def map(self, x):
        return (x-self.mu) / self.sigma

class AnatomyFeaturemapper(object):
    def __init__(self, vectorizer, whitener=None):
        self.vectorizer = vectorizer
        self.whitener = whitener

    def map(self, data):
        x, y = self.vectorizer.map2(data)
        if self.whitener:
            x = self.whitener.map(x)
        return x, y

    def mapX(self, data):
        x = self.vectorizer.mapX(data)
        if self.whitener:
            x = self.whitener.map(x)
        return x

    def getLabels(self, predictions):
        result = []
        for i in range(len(predictions)):
            result.append(self.vectorizer.labelMapper.inverseMap(predictions[i]))
        return result

    def getAllLabels():
        possibleValues = 17 # Support 17 labels.
        result = []
        for i in range(possibleValues):
            result.append({'Label' : str(i)})
        return result

    def buildVectorizer(data):
        # Using all the raw data we should build featuremappers
        topModalities = fm.selectTopWords(data, lambda x: x['Modality'], len(data))
        modalityMapper = fm.DictionaryFeatureMap2(topModalities, lambda x: x['Modality'])
        codeMapper = fm.CodePartFeatureMap2(data, lambda x: x['Code'], lambda x: x[0:min(len(x), 3)])
        topBodypartWords = fm.selectTopWords(data, lambda x: x['BodyPart'], 100)
        bodypartMapper = fm.DictionaryFeatureMap2(topBodypartWords, lambda x: x['BodyPart'])
        topDescriptionWords = fm.selectTopWords(data, lambda x: x['Description'], 100) 
        descriptionMapper = fm.DictionaryFeatureMap2(topDescriptionWords, lambda x: x['Description'])
        
        # 
        labelMapper = fm.DictionaryLabelMap2(AnatomyFeaturemapper.getAllLabels(), lambda x: str(x['Label']))

        return fm.mapper([modalityMapper,codeMapper, bodypartMapper, descriptionMapper], labelMapper)

    def buildWhitener(x):
        mu = np.mean(x, axis = 0)
        sigma = np.std(x, axis = 0) + 1e-5
        return AnatomyWhitener(mu, sigma)

    def build(data):
        pipe = AnatomyFeaturemapper.buildVectorizer(data)
        x, y = pipe.map2(data)
        whitener = AnatomyFeaturemapper.buildWhitener(x)
        x = whitener.map(x)
        mapper = AnatomyFeaturemapper(pipe, whitener)
        return mapper, x, y


class AnatomyModel(object):
    def __init__(self, featureMapper):
        inputDimension = featureMapper.vectorizer.getDimension()
        outputDimension = featureMapper.vectorizer.labelMapper.getRange()
        self.x = T.matrix('x')
        self.y = T.ivector('y')            
        self.classifier = nnlayer.MLP(rng=np.random.RandomState(1234), input = self.x, topology = [inputDimension, 100, outputDimension])
        self.cost = self.classifier.negative_log_likelihood(self.y)
        self.costFunction = (self.classifier.params, self.cost)

    
    def makeShared(self, xData, yData):
        x_shared = theano.shared(xData, borrow=True)
        y_shared = T.cast(theano.shared(yData, borrow=True), 'int32')
        return x_shared, y_shared
                    
    def train(self, xTrain, yTrain):
        train_x, train_y = self.makeShared(xTrain, yTrain)
        trainer = Trainer.MLPBatchTrainer()
        epochCosts = trainer.train(self.x, self.y, self.costFunction, train_x, train_y, batch_size=512, learning_rate = 0.001, epochs = 10, rms=True)

    def test(self, xTest, yTest):
        test_x, test_y = self.makeShared(xTest, yTest)
        batch_size=1
        index = T.lscalar()  # index to a [mini]batch
        test_model = theano.function(inputs=[index],
                outputs=self.classifier.errors(self.y),
                givens={
                    self.x: test_x[index * batch_size: (index + 1) * batch_size],
                    self.y: test_y[index * batch_size: (index + 1) * batch_size]})

        n_test_batch = int(test_x.get_value(borrow=True).shape[0] / batch_size)
        errorVector = [test_model(i) for i in range(n_test_batch)]
        print("Avg. error {0}".format(np.mean(errorVector)))

    def predict(self, xPredict):
        x_predict = theano.shared(xPredict, borrow=True)
        predict_model = theano.function(inputs=[],
                                        outputs=self.classifier.y_pred,
                                        givens = {self.x : x_predict})
        return predict_model()


class TestClass(object):
    def __init__(self):
        print("Construction")

        
    def split(self, data, fraction = 1.0):
        trainLength = int(len(data)*fraction)        
        return data[:trainLength], data[trainLength:]
    
    
    def createModel(self, listOfData):        
        trainData, testData = self.split(listOfData, fraction=1.0)
        
        #Store the mapping used.
        self.featureMapper, xTrain, yTrain = AnatomyFeaturemapper.build(trainData)
        self.model = AnatomyModel(self.featureMapper)
        
        #Train the model
        epochCosts = self.model.train(xTrain, yTrain)
        
        # Create mapped test data.
        #xTest, yTest = self.featureMapper.map(testData)
        #self.model.test(xTest, yTest)


    def updateModel(self, listOfData):
        xTrain, yTrain = self.featureMapper.map(listOfData)
        #Train the model
        epochCosts = self.model.train(xTrain, yTrain)

    def predict(self, data):
        xPredict = self.featureMapper.mapX(data)
        yPredict = self.model.predict(xPredict)
        result = self.featureMapper.getLabels(yPredict)
        return result


