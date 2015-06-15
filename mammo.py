import os
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from Layers import LogisticRegression
from Layers import nnlayer
from Mammo.MammoData import MammoData
from Mammo.DataUtils import loadData, loadPatchData, flatten
from Trainer.Trainer import MLPBatchTrainer, VariableAndData
from Trainer.Persistence import PersistenceManager
import plotutils

import pickle


def make_shared(data, size):
    # Prepare validation
    dataX, dataY = flatten(data, size)
    data_x = theano.shared(dataX, borrow=True)
    data_y = theano.shared(dataY, borrow=True)
    return data_x, data_y


def testModel(modelFilename, testData, size):
    

    # Create shared data
    test_x, test_y = make_shared(testData, size)
    
    # Load the model
    persistence = PersistenceManager()
    persistence.set_filename(modelFilename)
    x, y, classifier = persistence.load_model()
    
    # Create prediction function.
    pred_func = theano.function(inputs = [],
                                outputs = [classifier.y_pred, classifier.errors(y)],
                                givens = {x:test_x, y:test_y})    


    preds = pred_func()
    print("prediction done")

    plt.set_cmap('gray')    
    for i in range(len(testData)):
        s = testData[i]
        print("Dist: {0}".format(preds[1][i]))        
        normalizedCoords = preds[0][i]
        px = np.round(normalizedCoords[0] * s.width)
        py = np.round(normalizedCoords[1] * s.height)

        pxCopy = s.getAnnotated()
        pxCopy[py, px] = 0.85*np.max(pxCopy)

        plt.imshow(pxCopy)
        plt.show()    




def pretrainConv(trainData, nFilters, shape):
    # Randomly sample patches from the trainData
    # Each patch should be of size shape.
    nSamples = 50000
    data_x = np.zeros((nSamples, shape[0]*shape[1]), dtype = 'float32')
    
    rng =  np.random.RandomState(4321)
    maxIndex = len(trainData) - 1
    # Assuming that all samples have the same shape.
    
    halfY = np.ceil(shape[1]/2)
    minY = halfY
    maxY = trainData[0].height - halfY
    halfX = np.ceil(shape[0]/2) 
    minX = halfX
    maxX = trainData[0].width - halfX

    imageIndexes = rng.uniform(0, maxIndex, size =(nSamples)) 
    xPositions = rng.uniform(minX, maxX, size=(nSamples))
    yPositions = rng.uniform(minY, maxY, size=(nSamples))
    for idx in range(nSamples):
        i = int(imageIndexes[idx])
        x = int(xPositions[idx])
        y = int(yPositions[idx])
        mammoSample = trainData[i]
        xStart = x - halfX
        yStart = y - halfY
        xEnd = xStart + shape[0]
        yEnd = yStart + shape[1]
        data_x[idx] = mammoSample.pixelData[yStart:yEnd, xStart:xEnd].ravel()
        #plt.gray()
        #plt.imshow(data_x[idx].reshape((shape[0], shape[1])))
        #plt.show()

        
    
    # Prepare testdata.
    nTrain = nSamples * 0.95
    xTrain = data_x[:nTrain]
    xValidation = data_x[nTrain:]
    train_x = theano.shared(xTrain, borrow=True)
    validation_x = theano.shared(xValidation, borrow=True)
    
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are (x,y) coordinates.


    layer = nnlayer.ReluLayer(rng, x, shape[0]*shape[1], nFilters)       

    tt = MLPBatchTrainer()        
    
    #pre-training
    dae = nnlayer.DAE(rng, layer, 0.2)
    pretrainFunction, preStateManager = tt.getEpochTrainer((dae.params, dae.cost), [VariableAndData(x, train_x)], batch_size = 64, rms = True)
    preTrainValid = theano.function(inputs = [],
                            outputs = [dae.cost],
                            givens = {x:validation_x})
    preStats = tt.trainALR(pretrainFunction, 
                           preTrainValid, 
                           initial_learning_rate = 0.03, 
                           epochs = 5, 
                           #max_runs = 25,
                           max_runs = 1,
                           state_manager = preStateManager)
    pre_scores = [item["validation_score"] for item in preStats]
    plt.plot(pre_scores)
    plt.show()
    W = layer.W.get_value()
    b = layer.b.get_value()
    plotutils.plot_columns(W, (shape[0], shape[1]))    
    with open(r'.\SavedModels\conv_filters.pkl', 'wb') as f:
        pickle.dump((W, b), f)



def trainModel(modelFilename, trainData, validationData, size):
    
    #pretrainConv(trainData, 10, (5, 5))
    with open(r'.\SavedModels\conv_filters.pkl', 'rb') as f:
        W_conv, b_conv = pickle.load(f)

    # Prepare testdata.
    train_x, train_y = make_shared(trainData, size)
    validation_x, validation_y = make_shared(validationData, size)


    rng = np.random.RandomState(1234)
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are (x,y) coordinates.


    batch_size = 512
    tst = nnlayer.ConvNet(rng, x, (50, 50), [(10, 5, 5)], rectified=True)

    #plotutils.plot_columns(W_conv, (5, 5))
    W_t = np.zeros((10, 1, 5, 5), dtype='float32')
    for i in range(10):
        filt = W_conv[:, i].reshape((5, 5))
        filt =  np.fliplr(filt)
        filt = np.flipud(filt)
        W_t[i, 0] = filt
    
    #plotutils.plot_tensor_image(W_t)

    tst.layers[0].set_params(W_t, b_conv)







    Wbefore = tst.layers[0].W.get_value()
    
    #plotutils.plot_tensor_image(Wbefore)

    print("Size {0}".format(tst.output_size))
    input_dimension = tst.output_size#size**2
    output_dimension = 2
    classifier = nnlayer.MLPReg(rng=rng, 
                                input=tst.output, 
                                topology=[(input_dimension,), 
                                          (10, nnlayer.ReluLayer), 
                                          (output_dimension, nnlayer.LinearRegressionLayer)])   
    #classifier = nnlayer.MLPReg(rng=rng, input=tst.output, topology=[input_dimension, output_dimension], rectified=True, dropout_rate=0.5)   
    cost =  classifier.cost(y) #+ 0.003*classifier.L2_sqr #+ 0.003*tst.L2_sqr
    costParams = []    
    costParams.extend(tst.params)
    costParams.extend(classifier.params)
    costFunction = (costParams, cost)


    tt = MLPBatchTrainer()        

    #pre-training
    #dae = nnlayer.DAE(rng, classifier.hiddenLayers[0], 0.2)
    #pretrainFunction, preStateManager = tt.getEpochTrainer((dae.params, dae.cost), [VariableAndData(x, train_x)], batch_size = 64, rms = True)
    #preTrainValid = theano.function(inputs = [],
    #                        outputs = [dae.cost],
    #                        givens = {x:validation_x})
    #preStats = tt.trainALR(pretrainFunction, 
    #                       preTrainValid, 
    #                       initial_learning_rate = 0.001, 
    #                       epochs = 5, 
    #                       max_runs = 5,
    #                       state_manager = preStateManager)
    #pre_scores = [item["validation_score"] for item in preStats]
    #plt.plot(pre_scores)
    #plt.show()
    #plotutils.plot_columns(classifier.hiddenLayers[0].W.get_value(), (50, 50))


   
    
    # Supervised training
    valid_func = theano.function(inputs = [],
                            outputs = [classifier.cost(y)],
                            givens = {x:validation_x, y:validation_y})                            

    variableAndData = (VariableAndData(x, train_x), VariableAndData(y, train_y))
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

    Wafter = tst.layers[0].W.get_value()
    plotutils.plot_tensor_image(Wafter)

    # Save model to disk
    persistence = PersistenceManager()
    persistence.set_filename(modelFilename)
    persistence.set_model(x, y, classifier)
    persistence.save_model()




def main():
    size = 50
    filename= r".\SavedModels\test_model.pkl"
    trainData, testData = loadData(size, split=0.95)
    print("Data loaded")

    trainModel(filename, trainData, testData, size)
    testModel(filename, testData, size)

    #handleHiResData()

    #sample = MammoData()
    #sample.load(r'G:\temp\conv_net_mammo_hires\a68cec1a-2255-46e4-934d-18f87802e779')

    #ctx = MammoContext()
    #ctx.createModel()
    #res = ctx.internal_predict(sample)


        
if __name__ == "__main__":
    main()
