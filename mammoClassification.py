import os
import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from Layers import LogisticRegression
from Layers import nnlayer
from Mammo.MammoData import MammoData
import Mammo.DataUtils as utils
from Trainer.Trainer import MLPBatchTrainer, VariableAndData
from Trainer.Persistence import PersistenceManager
import plotutils

import pickle



def compute_ZCA(X):
    
    # Compute per-patch mean value.
    mu = np.mean(X, axis = 1)
    # X is now transposed.
    X = (X.T - mu)

    sigma = np.dot(X, X.T) / X.shape[1]

    # Compute SVD
    U, S, V = np.linalg.svd(sigma)
    S_norm = np.diag(1./np.sqrt(S + 1e-5))
    
    # Compute whitening matrix (and transpose it)
    Wmat = np.dot(U, np.dot(S_norm, U.T))    
    # Return a matrix that can be applied on
    # data batches with one sample per row.
    return (Wmat.T).astype('float32')

def apply_ZCA(X, Wmat):
    X = (X.T -  np.mean(X, axis = 1)).T
    return np.dot(X, Wmat)

def preprocess_data(data, size, Wmat = None):
    # Prepare validation
    dataX, dataY = utils.flattenClassification(data, size)    
    if not (Wmat is None):
        dataX = apply_ZCA(dataX, Wmat)
    return dataX, dataY


def make_shared(data, size, Wmat = None):
    # Prepare validation
    dataX, dataY = preprocess_data(data, size, Wmat)    
    data_x = theano.shared(dataX, borrow=True)
    data_y = theano.tensor.cast(theano.shared(dataY, borrow=True), dtype='int32')
    return data_x, data_y

def trainModel(modelFilename, allTrainDataImage, patchSize, margin, Wmat=None):    
    
    # Split into train and validation.
    trainDataImage, validationDataImage = utils.splitData(allTrainDataImage, 0.9)
    trainData = utils.makeClassificationPatches(trainDataImage, patchSize, margin)    
    validationData = utils.makeClassificationPatches(validationDataImage, patchSize, margin)
    
    trainDataImage = None
    validationDataImage = None
    allTrainDataImage = None
    
    print("Training data size {0}, Validation data size {1}".format(len(trainData), len(validationData)))
    
    # Create shared
    train_x, train_y = make_shared(trainData, patchSize, Wmat)            

    rng = np.random.RandomState(1234)
    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are 0-1 labels.

    input_dimension = patchSize**2
    output_dimension = 2
    classifier = nnlayer.MLPReg(rng=rng, input=x, topology=[(input_dimension,),
                                                            (500, nnlayer.ReluLayer),                                                           
                                                            (output_dimension, nnlayer.LogisticRegressionLayer)])

    cost = classifier.cost(y) #+ 0.0003*classifier.L2_sqr
    costParams = []
    costParams.extend(classifier.params)
    costFunction = (costParams, cost)

    tt = MLPBatchTrainer()

    # Create shared
    validation_x, validation_y = make_shared(validationData, patchSize, Wmat)
    valid_func = theano.function(inputs = [],
                            outputs = [T.mean(classifier.errors(y))],
                            #outputs = [cost],
                            #outputs = [classifier.cost(y)],
                            givens = {x:validation_x, y:validation_y})                            

    variableAndData = (VariableAndData(x, train_x), VariableAndData(y, train_y, size=len(trainData)))
    epochFunction, stateMananger = tt.getEpochTrainer(costFunction, 
                                                      variableAndData, 
                                                      batch_size=64, 
                                                      rms = True, 
                                                      momentum=0.9,
                                                      randomize=True)        
        
    # Train with adaptive learning rate.
    stats = tt.trainALR(epochFunction, 
                        valid_func, 
                        initial_learning_rate=0.001, 
                        epochs=3, 
                        convergence_criteria=0.0001, 
                        max_runs=50,
                        state_manager = stateMananger)

    validation_scores = [item["validation_score"] for item in stats]
    train_scorees = [item["training_costs"][-1] for item in stats]
    #train_scorees = stats[0]["training_costs"]
    plt.plot(validation_scores, 'g')
    plt.plot(train_scorees, 'r')
    plt.show()


    e_func = theano.function(inputs = [],
                            outputs = [classifier.errors(y)],
                            #outputs = [classifier.cost(y)],
                            givens = {x:validation_x, y:validation_y})                            

    print("avg error: {0}".format(np.mean(e_func())))


    mgr =  PersistenceManager()
    mgr.set_filename(modelFilename)
    mgr.set_model(x, y, classifier)
    mgr.save_model()

def testModel(modelFilename, testDataImage, patchSize, margin, Wmat = None):

       
    mgr = PersistenceManager()
    mgr.set_filename(modelFilename)
    x, y, classifier = mgr.load_model()

    def testPatches():
        testData = utils.makeClassificationPatches(testDataImage, patchSize, margin)
        test_x, test_y = make_shared(testData, patchSize, Wmat)
        idx = T.lscalar()
        test_func = theano.function(inputs = [idx],
                                outputs = classifier.errors(y),
                                givens = {x:test_x[idx:idx+1], y:test_y[idx:idx+1]})

        errorVector = [test_func(i) for i in range(len(testData))]
        mean_error = np.mean(errorVector)
        sum_error = np.sum(errorVector)
        print("#Samples: {0} Average error: {1} #Errors: {2}".format(len(testData), mean_error, sum_error))

        posCnt = 0
        for e, sample in zip(errorVector, testData):
            msg = "Correct"
            if e == 1:
                if sample.hasCircle:
                    posCnt += 1
                #print("Is nipple: {0}".format(sample.hasCircle))
                #sample.show()
                msg = "FAIL"        
            #print("{0}: Is nipple: {1}".format(msg, sample.hasCircle))
            #sample.show()
        print("#Pos error: {0}, avg: {1}".format(posCnt, posCnt / len(testData)))
        
    # Call the patch tests.        
    #testPatches()
    
    
    def testConv():
        # Create prediction function
        test_func = theano.function(inputs = [x],
                            outputs = classifier.outputLayer.p_y_given_x[:, 1])

        for testImage in testDataImage:    
            convPatches, positions = utils.makeConvData(testImage, patchSize, stride=margin)
            #test_x, test_y = make_shared(convPatches, patchSize)    
            test_x, test_y = utils.flattenClassification(convPatches, patchSize)
            if not (Wmat is None):
                test_x = apply_ZCA(test_x, Wmat)

            

            predictions = test_func(test_x)
            heat_map = np.zeros_like(testImage.pixelData)
            avg_map = np.zeros_like(testImage.pixelData) + 1e-5
            whitened = np.zeros_like(testImage.pixelData)
            pCnt = 0
            errors = 0
            posError = 0
            posCount = 0            
            idx = 0
            for pred, pos, patch in zip(predictions, positions, convPatches):
                error = 0
                if patch.circle_inside_margin(margin):
                    posCount += 1
                    if pred < 0.5:
                        error = 1
                        posError += 1
                    
                else:
                    if pred >= 0.5:
                        error = 1
                
                errors += error
                pCnt += 1
                xStart = pos[0] - patchSize / 2
                yStart = pos[1] - patchSize / 2
                #heat_map[yStart:yStart + patchSize, xStart:xStart + patchSize] += pred >= 0.5
                heat_map[yStart:yStart + patchSize, xStart:xStart + patchSize] += 1 if pred >= 0.5 else 0
                avg_map[yStart:yStart + patchSize, xStart:xStart + patchSize] += 1
                whitened[yStart:yStart + patchSize, xStart:xStart + patchSize] = test_x[idx, :].reshape(patchSize, patchSize)
                idx += 1

            avg_error = errors / pCnt
            avg_pos_error = posError / posCount
            print("Avg error: {0}, Positive error: {1}".format(avg_error, avg_pos_error))
            # Treat as distribution by using sum as partition func.
            heat_map = heat_map / np.sum(heat_map)#avg_map                                
            print("Max: {0}".format(np.max(heat_map)))
            heat_map /= np.max(heat_map)

            cutoff = 0.5*np.max(heat_map)
            heat_map -= cutoff
            heat_map = np.clip(heat_map, 0, 1.0)
            heat_map /= 1-cutoff
            
                        

            #idxes = heat_map > 0
            #anot = np.copy(testImage.pixelData)
            #anot[idxes] = np.max(anot)
            test_x[:, :] = 0

            #plt.set_cmap('hot')
            f, axarr = plt.subplots(2)
            im = axarr[0].imshow(heat_map +  testImage.pixelData/2, cmap='hot')
            #im = axarr[0].imshow(whitened, cmap='gray')
            im = axarr[1].imshow(testImage.pixelData, cmap='gray')
            #plt.gray()
            #plt.imshow(heat_map)
            #plt.imshow(heat_map*testImage.pixelData)
            #plt.imshow(anot)
            #plt.imshow(heat_map + 0.2*testImage.pixelData)
            plt.show()
            #testImage.show()
    # Run the convolutional tests.
    testConv()        


def prepare_ZCA(filename, data, patchSize, margin):
    # Prepare validation
    data = utils.makeClassificationPatches(data, patchSize, margin) 
    dataX, dataY = utils.flattenClassification(data, patchSize)    
    
    Wmat = compute_ZCA(dataX)
    with open(filename, 'wb') as f:
        pickle.dump(Wmat, f)

def load_ZCA(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    patchSize = 16
    margin = 4
    #makeClassificationData()
    modelFilename = r".\SavedModels\test_classification_model.pkl"
    zcaFilename = r".\SavedModels\ZCA.pkl"    
    trainDataImage, testDataImage = utils.loadClassificationData(split=0.9)    
    
    # prepare a whitening matrix.
    #prepare_ZCA(zcaFilename, trainDataImage, patchSize, margin)
    Wmat = load_ZCA(zcaFilename)

    #trainModel(modelFilename, trainDataImage, patchSize, margin, Wmat)
    testModel(modelFilename, testDataImage, patchSize, margin, Wmat)





    #sample = MammoData()
    #sample.load(r'G:\temp\conv_net_mammo_hires\a68cec1a-2255-46e4-934d-18f87802e779')

    #ctx = MammoContext()
    #ctx.createModel()
    #res = ctx.internal_predict(sample)


        
if __name__ == "__main__":
    main()
