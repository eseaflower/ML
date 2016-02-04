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
import lasagne



def compute_MU(X):    
    # Compute per-feature mean value.
    return np.mean(X, axis = 0).astype('float32')

def apply_MU(X, mu):    
    return X - mu

def preprocess_data(data, size, mu = None):
    # Prepare validation
    dataX, dataY = utils.flattenClassification(data, size)    
    if not (mu is None):
        dataX = apply_MU(dataX, mu)
    return dataX, dataY


def make_shared(data, size, mu = None):
    # Prepare validation
    dataX, dataY = preprocess_data(data, size, mu)    
    data_x = theano.shared(dataX, borrow=True)
    data_y = theano.tensor.cast(theano.shared(dataY, borrow=True), dtype='int32')
    return data_x, data_y

def trainModel(modelFilename, allTrainDataImage, patchSize, margin, mu=None):    
    
    # Split into train and validation.
    trainDataImage, validationDataImage = utils.splitData(allTrainDataImage, 0.9)
    trainData = utils.makeClassificationPatches(trainDataImage, patchSize, margin)    
    validationData = utils.makeClassificationPatches(validationDataImage, patchSize, margin)
    
    trainDataImage = None
    validationDataImage = None
    allTrainDataImage = None
    
    print("Training data size {0}, Validation data size {1}".format(len(trainData), len(validationData)))
    
    # Create shared
    train_x, train_y = make_shared(trainData, patchSize, mu)            

    rng = np.random.RandomState(1234)
    lasagne.random.set_rng(rng)

    # allocate symbolic variables for the data
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are 0-1 labels.

    input_dimension = patchSize**2
    output_dimension = 2

    


    #classifier = nnlayer.MLPReg(rng=rng, input=x, topology=[(input_dimension,),
    #                                                        (500, nnlayer.ReluLayer),                                                           
    #                                                        (output_dimension, nnlayer.LogisticRegressionLayer)])
    classifier = nnlayer.ClassificationNet(input=x, topology=[(input_dimension,),
                                                       (nnlayer.LasangeNet.Reshape, (-1, 1, patchSize, patchSize)),
                                                       (nnlayer.LasangeNet.Conv, 32, 3, {'pad':'same'}),
                                                       (nnlayer.LasangeNet.Pool,),
                                                       (nnlayer.LasangeNet.Conv, 64, 3, {'pad':'same'}),
                                                       (nnlayer.LasangeNet.Pool,),
                                                              #(nnlayer.LasangeNet.DropoutLayer, 0.2),
                                                       #(nnlayer.LasangeNet.BatchNorm, nnlayer.LasangeNet.ReluLayer, 500),
                                                       #(nnlayer.LasangeNet.ReluLayer, 500),
                                                       #(nnlayer.LasangeNet.DropoutLayer, ),
                                                       #(nnlayer.LasangeNet.ReluLayer, 500),
                                                       #(nnlayer.LasangeNet.DropoutLayer, ),
                                                       (nnlayer.LasangeNet.SoftmaxLayer, output_dimension)])


    cost = classifier.cost(y) #+ 0.0003*classifier.L2
    costParams = []
    costParams.extend(classifier.params)
    costFunction = (costParams, cost, classifier.accuracy(y))

    tt = MLPBatchTrainer()

    # Create shared
    validation_x, validation_y = make_shared(validationData, patchSize, mu)
    
    v_start = T.iscalar()
    v_end = T.iscalar()
    bv_func = theano.function(inputs = [v_start, v_end],
                        outputs = [classifier.validation_cost(y), classifier.accuracy(y)],
                        givens = {x:validation_x[v_start:v_end], y:validation_y[v_start:v_end]})                            

    def batch_validation():
        maxIdx = len(validationData)
        nv_batches =  maxIdx // 128
        tc = [0, 0]
        for i in range(nv_batches):
            d_start = min(i*128, maxIdx)
            d_end = min((i + 1)*128, maxIdx)
            bc = bv_func(d_start, d_end)
            factor = (d_end - d_start) / 128.0
            tc[0] += bc[0]*factor
            tc[1] += bc[1]*factor
        tc[0] /= nv_batches
        tc[1] /= nv_batches
        return tc
    variableAndData = (VariableAndData(x, train_x), VariableAndData(y, train_y, size=len(trainData)))
    epochFunction, stateMananger = tt.getEpochTrainer(costFunction, 
                                                      variableAndData, 
                                                      batch_size=128, 
                                                      rms = True, 
                                                      momentum=0.9,
                                                      randomize=True,
                                                      updateFunction=MLPBatchTrainer.wrapUpdate(lasagne.updates.rmsprop))        
        
    # Train with adaptive learning rate.
    stats = tt.trainALR2(epochFunction, 
                        #valid_func, 
                        batch_validation,
                        #initial_learning_rate=0.001, 
                        initial_learning_rate=0.01, 
                        max_runs=100,
                        state_manager = stateMananger)

    validation_scores = [item["validation_outputs"][1] for item in stats]
    train_scorees = [item["training_outputs"][1] for item in stats]    
    plt.plot(validation_scores, 'g')
    plt.plot(train_scorees, 'r')
    plt.show()


    #e_func = theano.function(inputs = [],
    #                        outputs = classifier.accuracy(y),
    #                        #outputs = [classifier.cost(y)],
    #                        givens = {x:validation_x, y:validation_y})                            

    ##print("avg error: {0}".format(np.mean(e_func())))
    #print("validation accuracy: {0}".format(e_func()))


    mgr =  PersistenceManager()
    mgr.set_filename(modelFilename)
    mgr.set_model(x, y, classifier)
    mgr.save_model()

def testModel(modelFilename, testDataImage, patchSize, margin, mu = None):

       
    mgr = PersistenceManager()
    mgr.set_filename(modelFilename)
    x, y, classifier = mgr.load_model()

    conv_layer = classifier.layers[2]
    #W = conv_layer.W.get_value(borrow=False)
    #plotutils.plot_tensor_image(W)



    def testBackward():        
        #x_s = theano.shared(np.random.normal(0.0, 0.1, size=(1,patchSize*patchSize)).astype('float32'))
        x_s = theano.shared(np.zeros((1, patchSize*patchSize), dtype='float32'))
        y_s = theano.shared(np.ones((1,), dtype='int32'))
        c = classifier.validation_cost(y) + 0.01*T.sum(abs(x))
        loss = theano.clone(c, {x:x_s, y:y_s})                           
        upd = lasagne.updates.rmsprop(loss, [x_s], learning_rate=0.01)
        func = theano.function(inputs=[],
                               outputs = loss,
                               updates = upd)



        for i in range(10000):
            res = func()
            print("Loss: {0}".format(res))
            #if i%100 == 0:
            #    img = x_s.get_value(borrow=False)
            #    img = img.reshape((patchSize, patchSize))
            #    plt.imshow(img, cmap='gray')
            #    plt.show()            

        img = x_s.get_value(borrow=False)
        img = img.reshape((patchSize, patchSize))
        plt.imshow(img, cmap='gray')
        plt.show()            


    #testBackward()



    def testPatches():
        testData = utils.makeClassificationPatches(testDataImage, patchSize, margin)
        test_x, test_y = make_shared(testData, patchSize, mu)
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
    testPatches()
    
    
    def testConv():
        # Create prediction function
        #test_func = theano.function(inputs = [x],
        #                    outputs = classifier.outputLayer.p_y_given_x[:, 1])

        test_func = theano.function(inputs = [x],
                                    outputs = classifier.predict_output[:, 1])
        
        c_out = lasagne.layers.get_output(conv_layer, x, deterministic=True)
        
        c_out_func = theano.function(inputs = [x],
                                    outputs = c_out)



        for testImage in testDataImage:    
            convPatches, positions = utils.makeConvData(testImage, patchSize, stride=margin)
            #test_x, test_y = make_shared(convPatches, patchSize)    
            test_x, test_y = utils.flattenClassification(convPatches, patchSize)
            if not (mu is None):
                test_x = apply_MU(test_x, mu)

            

            predictions = test_func(test_x)
            conv_res = c_out_func(test_x)            

            c_img = np.zeros((32, testImage.pixelData.shape[0], testImage.pixelData.shape[1]))
            heat_map = np.zeros_like(testImage.pixelData)
            avg_map = np.zeros_like(testImage.pixelData) + 1e-5
            whitened = np.zeros_like(testImage.pixelData)
            pCnt = 0
            errors = 0
            posError = 0
            posCount = 0            
            idx = 0
            for pred, pos, patch, c_p in zip(predictions, positions, convPatches, conv_res):
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
                xStart = int(pos[0] - patchSize / 2)
                yStart = int(pos[1] - patchSize / 2)
                #heat_map[yStart:yStart + patchSize, xStart:xStart + patchSize] += pred >= 0.5
                heat_map[yStart:yStart + patchSize, xStart:xStart + patchSize] += 1 if pred >= 0.5 else 0
                avg_map[yStart:yStart + patchSize, xStart:xStart + patchSize] += 1
                whitened[yStart:yStart + patchSize, xStart:xStart + patchSize] = test_x[idx, :].reshape(patchSize, patchSize)
                c_img[:, yStart:yStart + 14, xStart:xStart + 14] = c_p[:, :]
                idx += 1

            
            #for i in range(32):
            #    plt.imshow(c_img[i])
            #    plt.show()

            avg_error = errors / pCnt
            avg_pos_error = posError / posCount
            print("Avg error: {0}, Positive error: {1}".format(avg_error, avg_pos_error))
            # Treat as distribution by using sum as partition func.
            heat_map = heat_map / np.sum(heat_map)#avg_map                                
            print("Max: {0}".format(np.max(heat_map)))
            heat_map /= np.max(heat_map)

            cutoff = 0.75*np.max(heat_map)
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


def prepare_MU(filename, data, patchSize, margin):
    # Prepare validation
    data = utils.makeClassificationPatches(data, patchSize, margin) 
    dataX, dataY = utils.flattenClassification(data, patchSize)    
    
    mu = compute_MU(dataX)
    with open(filename, 'wb') as f:
        pickle.dump(mu, f)

def load_MU(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main():
    patchSize = 32
    margin = 4
    #makeClassificationData()
    modelFilename = r".\SavedModels\test_classification_model.pkl"
    MUFilename = r".\SavedModels\MU.pkl"        
    trainDataImage, testDataImage = utils.loadClassificationData(split=0.9)    
    
    # prepare a whitening matrix.
    prepare_MU(MUFilename, trainDataImage, patchSize, margin)
    mu = load_MU(MUFilename)

    trainModel(modelFilename, trainDataImage, patchSize, margin, mu)
    #testModel(modelFilename, testDataImage, patchSize, margin, mu)


    #sample = MammoData()
    #sample.load(r'G:\temp\conv_net_mammo_hires\a68cec1a-2255-46e4-934d-18f87802e779')

    #ctx = MammoContext()
    #ctx.createModel()
    #res = ctx.internal_predict(sample)


        
if __name__ == "__main__":
    main()
