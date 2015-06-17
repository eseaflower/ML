import os
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from Layers import LogisticRegression
from Layers import nnlayer
from Mammo.MammoData import MammoData
import Mammo.DataUtils as utils
#from Mammo.DataUtils import  makeClassificationData, loadClassificationData, makeClassificationPatches, splitData
from Trainer.Trainer import MLPBatchTrainer, VariableAndData
from Trainer.Persistence import PersistenceManager
import plotutils

import pickle


def make_shared(data, size):
    # Prepare validation
    dataX, dataY = utils.flattenClassification(data, size)
    data_x = theano.shared(dataX, borrow=True)
    data_y = theano.tensor.cast(theano.shared(dataY, borrow=True), dtype='int32')
    return data_x, data_y


def trainModel(modelFilename, allTrainDataImage, patchSize):    
    
    # Split into train and validation.
    trainDataImage, validationDataImage = utils.splitData(allTrainDataImage, 0.9)
    trainData = utils.makeClassificationPatches(trainDataImage, patchSize)    
    validationData = utils.makeClassificationPatches(validationDataImage, patchSize)
    print("Training data size {0}, Validation data size {1}".format(len(trainData), len(validationData)))
    
    # Create shared
    train_x, train_y = make_shared(trainData, patchSize)        

    rng = np.random.RandomState(1234)
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are 0-1 labels.

    input_dimension = patchSize**2
    output_dimension = 2
    classifier = nnlayer.MLPReg(rng=rng, input=x, topology=[(input_dimension,),
                                                            (100, nnlayer.ReluLayer),
                                                           (output_dimension, nnlayer.LogisticRegressionLayer)])

    cost = classifier.cost(y) #+ 0.0001*classifier.L2_sqr
    costParams = []
    costParams.extend(classifier.params)
    costFunction = (costParams, cost)

    tt = MLPBatchTrainer()

    # Create shared
    validation_x, validation_y = make_shared(validationData, patchSize)
    valid_func = theano.function(inputs = [],
                            outputs = [classifier.cost(y)],
                            givens = {x:validation_x, y:validation_y})                            

    variableAndData = (VariableAndData(x, train_x), VariableAndData(y, train_y, size=len(trainData)))
    epochFunction, stateMananger = tt.getEpochTrainer(costFunction, variableAndData, batch_size=64, rms = True)        
        
    # Train with adaptive learning rate.
    stats = tt.trainALR(epochFunction, 
                        valid_func, 
                        initial_learning_rate=0.1, 
                        epochs=1, 
                        convergence_criteria=0.0001, 
                        max_runs=150,
                        state_manager = stateMananger)

    validation_scores = [item["validation_score"] for item in stats]
    #train_scorees = [item["training_costs"][-1] for item in stats]
    train_scorees = stats[0]["training_costs"]
    plt.plot(validation_scores, 'g')
    plt.plot(train_scorees, 'r')
    plt.show()


    mgr =  PersistenceManager()
    mgr.set_filename(modelFilename)
    mgr.set_model(x, y, classifier)
    mgr.save_model()



def testModel(modelFilename, testDataImage, patchSize):

    

    
    mgr = PersistenceManager()
    mgr.set_filename(modelFilename)
    x, y, classifier = mgr.load_model()

    def testPatches():
        testData = utils.makeClassificationPatches(testDataImage, patchSize)
        test_x, test_y = make_shared(testData, patchSize)
        idx = T.lscalar()
        test_func = theano.function(inputs = [idx],
                                outputs = classifier.errors(y),
                                givens = {x:test_x[idx:idx+1], y:test_y[idx:idx+1]})

        errorVector = [test_func(i) for i in range(len(testData))]
        mean_error = np.mean(errorVector)
        sum_error = np.sum(errorVector)
        print("#Samples: {0} Average error: {1} #Errors: {2}".format(len(testData), mean_error, sum_error))

        for e, sample in zip(errorVector, testData):
            msg = "Correct"
            if e == 1:
                msg = "FAIL"        
            print("{0}: Is nipple: {1}".format(msg, sample.hasCircle))
            sample.show()
    
    # Call the patch tests.        
    #testPatches()
    def testConv():
        # Create prediction function
        test_func = theano.function(inputs = [x],
                            outputs = classifier.outputLayer.p_y_given_x[:, 1])

        for testImage in testDataImage:    
            convPatches, positions = utils.makeConvData(testImage, patchSize, stride=patchSize-8)
            #test_x, test_y = make_shared(convPatches, patchSize)    
            test_x, test_y = utils.flattenClassification(convPatches, patchSize)
            predictions = test_func(test_x)
            heat_map = np.zeros_like(testImage.pixelData)
            avg_map = np.zeros_like(testImage.pixelData) + 1e-5
            pCnt = 0
            errors = 0
            posError = 0
            posCount = 0
            for pred, pos, patch in zip(predictions, positions, convPatches):
                if patch.hasCircle:
                    posCount += 1
                    if pred < 0.5:
                        errors += 1
                        posError += 1
                    
                else:
                    if pred >= 0.5:
                        errors += 1
                pCnt += 1
                xStart = pos[0] - patchSize / 2
                yStart = pos[1] - patchSize / 2
                heat_map[yStart:yStart + patchSize, xStart:xStart + patchSize] += pred
                avg_map[yStart:yStart + patchSize, xStart:xStart + patchSize] += 1

            avg_error = errors / pCnt
            avg_pos_error = posError / posCount
            print("Avg error: {0}, Positive error: {1}".format(avg_error, avg_pos_error))
            heat_map = heat_map / avg_map        
            plt.set_cmap('hot')
            #plt.imshow(heat_map*testImage.pixelData)
            plt.imshow(heat_map*testImage.pixelData)
            plt.show()
            #testImage.show()
    # Run the convolutional tests.
    testConv()        


def main():
    patchSize = 10

    #makeClassificationData()
    modelFilename = r".\SavedModels\test_classification_model.pkl"
    trainDataImage, testDataImage = utils.loadClassificationData(split=0.9)
    #trainModel(modelFilename, trainDataImage, patchSize)
    testModel(modelFilename, testDataImage, patchSize)





    #sample = MammoData()
    #sample.load(r'G:\temp\conv_net_mammo_hires\a68cec1a-2255-46e4-934d-18f87802e779')

    #ctx = MammoContext()
    #ctx.createModel()
    #res = ctx.internal_predict(sample)


        
if __name__ == "__main__":
    main()
