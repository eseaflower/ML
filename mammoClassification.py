import os
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from Layers import LogisticRegression
from Layers import nnlayer
from Mammo.MammoData import MammoData
from Mammo.DataUtils import loadData, loadPatchData, flattenClassification, makeClassificationData, loadClassificationData, makeClassificationPatches
from Trainer.Trainer import MLPBatchTrainer, VariableAndData
from Trainer.Persistence import PersistenceManager
import plotutils

import pickle


def make_shared(data, size):
    # Prepare validation
    dataX, dataY = flattenClassification(data, size)
    data_x = theano.shared(dataX, borrow=True)
    data_y = theano.tensor.cast(theano.shared(dataY, borrow=True), dtype='int32')
    return data_x, data_y


def trainModel(modelFilename, trainDataImage, validationDataImage):
    patchSize = 10
    trainData = makeClassificationPatches(trainDataImage, patchSize)    
    
    print("Size of training data {0}".format(len(trainData)))

    train_x, train_y = make_shared(trainData, patchSize)

    rng = np.random.RandomState(1234)
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are 0-1 labels.

    input_dimension = patchSize*2
    output_dimension = 2
    classifier = nnlayer.MLPReg(rng=rng, input=x, topology=[(input_dimension,),
                                                           (output_dimension, nnlayer.LogisticRegressionLayer)])

    cost = classifier.cost(y)
    costParams = []
    costParams.extend(classifier.params)
    costFunction = (costParams, cost)

    tt = MLPBatchTrainer()


    validationData = makeClassificationPatches(validationDataImage, patchSize)
    validation_x, validation_y = make_shared(validationData, patchSize)
    valid_func = theano.function(inputs = [],
                            outputs = [classifier.cost(y)],
                            givens = {x:validation_x, y:validation_y})                            

    variableAndData = (VariableAndData(x, train_x), VariableAndData(y, train_y, size=len(trainData)))
    epochFunction, stateMananger = tt.getEpochTrainer(costFunction, variableAndData, batch_size=64, rms = True)        
        
    # Train with adaptive learning rate.
    stats = tt.trainALR(epochFunction, 
                        valid_func, 
                        initial_learning_rate=0.0003, 
                        epochs=1, 
                        convergence_criteria=0.0001, 
                        max_runs=5,
                        state_manager = stateMananger)

    validation_scores = [item["validation_score"] for item in stats]
    #train_scorees = [item["training_costs"][-1] for item in stats]
    train_scorees = stats[0]["training_costs"]
    plt.plot(validation_scores, 'g')
    plt.plot(train_scorees, 'r')
    plt.show()





def main():

    #makeClassificationData()
    modelFilename = r".\SavedModels\test_classification_model.pkl"
    trainDataImage, testDataImage = loadClassificationData(split=0.9)
    trainModel(modelFilename, trainDataImage, testDataImage)






    #sample = MammoData()
    #sample.load(r'G:\temp\conv_net_mammo_hires\a68cec1a-2255-46e4-934d-18f87802e779')

    #ctx = MammoContext()
    #ctx.createModel()
    #res = ctx.internal_predict(sample)


        
if __name__ == "__main__":
    main()
