import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image
import scipy.misc
import glob
import pickle
import random
from Mammo.DataUtils import splitData
import theano
import theano.tensor as T
import lasagne
from Layers import nnlayer
from Trainer.Trainer import MLPBatchTrainer, VariableAndData
from Trainer.Persistence import PersistenceManager


rnd_state = np.random.RandomState(123)

PatDataDirectory = r'..\Data\Pat'

def generate_data(patchSize, samplesPerImage):
    
    def get_patches(data, patchSize, patchesPerImage):
        halfPatchSize = int(patchSize/2)
        minX = halfPatchSize
        maxX = data.shape[1] - halfPatchSize

        minY = halfPatchSize
        maxY = data.shape[0] - halfPatchSize
    
        xPositions = rnd_state.uniform(minX, maxX, size=(patchesPerImage,)).astype('int32')
        yPositions = rnd_state.uniform(minY, maxY, size=(patchesPerImage,)).astype('int32')
        result = []
        for x, y in zip(xPositions, yPositions):
            patch = data[y-halfPatchSize:y+halfPatchSize, x-halfPatchSize:x+halfPatchSize, :]
            result.append(patch)
        return result



    def patches_for_images(filepattern, patchSize, samplesPerImage):
        result = []
        for p in glob.glob(filepattern):
            data = matplotlib.image.imread(p)
            patches = get_patches(data, patchSize, samplesPerImage)
            result.extend(patches)
        return result        
    
    stroma_pattern = "{0}\\stroma*_cropped.png".format(PatDataDirectory)
    tumour_pattern = "{0}\\tumour*_cropped.png".format(PatDataDirectory)

        
    stroma_patches = patches_for_images(stroma_pattern, patchSize, samplesPerImage)
    tumour_patches = patches_for_images(tumour_pattern, patchSize, samplesPerImage)
    random.shuffle(stroma_patches, random=rnd_state.uniform)
    stroma_patches = np.asarray(stroma_patches)
    random.shuffle(tumour_patches, random=rnd_state.uniform)
    tumour_patches = np.asarray(tumour_patches)

    def visualize_samples():
        while True:
            si = rnd_state.uniform(0, len(stroma_patches), size=(1,)).astype('int32')
            plt.imshow(stroma_patches[si[0]])
            plt.show()
            ti = rnd_state.uniform(0, len(tumour_patches), size=(1,)).astype('int32')
            plt.imshow(tumour_patches[ti[0]])
            plt.show()
    #visualize_samples()        

    return stroma_patches, tumour_patches

def create_data(patchSize, samplesPerImage):
    stroma, tumour = generate_data(patchSize, samplesPerImage)
    stroma_filename = "{0}/stroma.pkl".format(PatDataDirectory)
    tumour_filename = "{0}/tumour.pkl".format(PatDataDirectory)
    with open(stroma_filename, "wb") as f:
        pickle.dump(stroma, f)
    with open(tumour_filename, "wb") as f:
        pickle.dump(tumour, f)

def load_data():
    stroma_filename = "{0}/stroma.pkl".format(PatDataDirectory)
    tumour_filename = "{0}/tumour.pkl".format(PatDataDirectory)
    with open(stroma_filename, "rb") as f:
        stroma = pickle.load(f)
    with open(tumour_filename, "rb") as f:
        tumour = pickle.load(f)
    return stroma, tumour



def merge_classes(negative, positive, shuffle=True):
    # Merge positive and negative
    numExamples = negative.shape[0] + positive.shape[0]
    trainX_shape = (numExamples,) + negative.shape[1:]
    
    data = np.zeros(trainX_shape, dtype = 'float32')
    targets = np.zeros((numExamples,), dtype='float32')
    data[:negative.shape[0]] = negative[:]
    targets[:negative.shape[0]] = 0
    data[negative.shape[0]:] = positive[:]
    targets[negative.shape[0]:] = 1
    if shuffle:
        permutaion =  rnd_state.permutation(data.shape[0])
        data = data[permutaion]
        targets = targets[permutaion]

    return data, targets

def get_data_sets():    
    stroma, tumour = load_data()
    train_stroma, test_stroma = splitData(stroma)
    train_tumour, test_tumour = splitData(tumour)
    trainX, trainY = merge_classes(train_stroma, train_tumour)
    testX, testY = merge_classes(test_stroma, test_tumour)
    return trainX, trainY, testX, testY


def get_shared_data(x, y):
    # Training data
    shared_x = theano.shared(x, borrow=True)
    shared_y = theano.tensor.cast(theano.shared(y, borrow=True), dtype='int32')
    return shared_x, shared_y


def train(trainX, trainY, validationX, validationY, modelFilename, patchSize, mu):    
    
    
    print("Train size: {0} validation size: {1}".format(trainX.shape[0], validationX.shape[0]))
    t_positive = np.sum(trainY > 0)
    print("Train - Positive: {0} fraction: {1}".format(t_positive, t_positive / trainY.shape[0]))
    v_positive = np.sum(validationY > 0)
    print("Validation - Positive: {0} fraction: {1}".format(v_positive, v_positive / validationY.shape[0]))
    
    #input("shg")
    
    trainX -= mu
    trainX, trainY = get_shared_data(trainX, trainY)
    
    rng = np.random.RandomState(1234)
    lasagne.random.set_rng(rng)


    # allocate symbolic variables for the data
    x = T.tensor4('x')  # the data is presented as batches of RGB images (batch, w, h, c)
    y = T.ivector('y')  # the labels are 0-1 labels.

    train_shape = trainX.get_value(borrow=True).shape
    input_dimension = (None,) + train_shape[1:]
    output_dimension = 2
    
    classifier = nnlayer.ClassificationNet(input=x, topology=[input_dimension,
                                                              #(nnlayer.LasangeNet.DropoutLayer, 0.5),                                                               
                                                              (nnlayer.LasangeNet.ReluLayer, 500),
                                                              (nnlayer.LasangeNet.DropoutLayer, ),                                                                
                                                              (nnlayer.LasangeNet.SoftmaxLayer, output_dimension)])


    cost = classifier.cost(y) + 0.01*classifier.L2
    costParams = []
    costParams.extend(classifier.params)
    costFunction = (costParams, cost, classifier.accuracy(y))

    tt = MLPBatchTrainer()

    # Create shared
    validation_shape = validationX.shape
    validationX -= mu
    validationX, validationY = get_shared_data(validationX, validationY)
    
    v_start = T.iscalar()
    v_end = T.iscalar()
    bv_func = theano.function(inputs = [v_start, v_end],
                        outputs = [classifier.validation_cost(y), classifier.accuracy(y)],
                        givens = {x:validationX[v_start:v_end], y:validationY[v_start:v_end]})                            

    def batch_validation():
        maxIdx = validation_shape[0]
        v_batch_size = min(maxIdx, 128)
        nv_batches = int(np.ceil(maxIdx / v_batch_size))
        tc = [0, 0]
        for i in range(nv_batches):
            d_start = min(i*v_batch_size, maxIdx)
            d_end = min((i + 1)*v_batch_size, maxIdx)
            bc = bv_func(d_start, d_end)
            factor = (d_end - d_start) / v_batch_size
            tc[0] += bc[0]*factor
            tc[1] += bc[1]*factor
        tc[0] /= nv_batches
        tc[1] /= nv_batches
        return tc
    variableAndData = (VariableAndData(x, trainX), VariableAndData(y, trainY, size=train_shape[0]))
    epochFunction, stateMananger = tt.getEpochTrainer(costFunction, 
                                                      variableAndData, 
                                                      batch_size=64, 
                                                      rms = True, 
                                                      momentum=0.9,
                                                      randomize=True,
                                                      updateFunction=MLPBatchTrainer.wrapUpdate(lasagne.updates.rmsprop))        
        
    # Train with adaptive learning rate.
    stats = tt.trainALR2(epochFunction, 
                        #valid_func, 
                        batch_validation,
                        #initial_learning_rate=0.001, 
                        initial_learning_rate=0.001, 
                        max_runs=10000,
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

def rnd_conv_patches(data, patchSize, nSamples):
    px = rnd_state.uniform(0, data.shape[1] - patchSize, size=(nSamples,)).astype('int32')
    py = rnd_state.uniform(0, data.shape[0] - patchSize, size=(nSamples,)).astype('int32')
    patches = []
    for x, y in zip(px, py):
        patch = data[y:y+patchSize, x:x+patchSize, :]
        patches.append(patch)
    return px, py, patches

def full_conv_patches(data, patchSize, step=1):
    px = np.arange(0, data.shape[1]-patchSize, step, dtype='int32')
    py = np.arange(0, data.shape[0]-patchSize, step, dtype='int32')
    xpos = []
    ypos = []
    patches = []
    for y in py:
        for x in px:    
            patch = data[y:y+patchSize, x:x+patchSize, :]
            patches.append(patch)
            xpos.append(x)
            ypos.append(y)
    return xpos, ypos, patches

def test(modelFilename, test_images, patchSize, mu):
    mgr = PersistenceManager()
    mgr.set_filename(modelFilename)
    x, y, classifier = mgr.load_model()
    
    eval_func = theano.function(inputs = [x],
                                outputs = classifier.predict_output[:, 1])
    
    for img in test_images:
        px, py, patches = full_conv_patches(img, patchSize, 2)
        testData = None
        testData = np.asarray(patches, dtype='float32')
        testData -= mu
        predictions = eval_func(testData)
            
        sample_map = np.zeros_like(img, dtype = 'float32')
        heat_map = np.zeros_like(img, dtype = 'float32')
        avg_map = np.zeros_like(img, dtype = 'float32') + 1e-5        
        idx = 0
        for pred, posX, posY in zip(predictions, px, py):
            value = pred if pred >= 0.5 else 1-pred
            heat_map[posY:posY + patchSize, posX:posX + patchSize] += [0, value, 0] if pred >= 0.5 else [value, 0, 0]
            avg_map[posY:posY + patchSize, posX:posX + patchSize] += 1                                    
            sample_map[posY:posY + patchSize, posX:posX + patchSize] = [0, 0, 1]

            
        # Treat as distribution by using sum as partition func.
        #heat_map = heat_map / np.sum(heat_map)#avg_map                                
        #print("Max: {0}".format(np.max(heat_map)))
        #heat_map /= np.max(heat_map)
        

        #cutoff = 0.0*np.max(heat_map)
        #heat_map -= cutoff
        heat_map /= np.max(heat_map, axis=2).reshape(heat_map.shape[0], heat_map.shape[1], 1)
        #heat_map = np.clip(heat_map, 0, 1.0)
        #heat_map /= 1-cutoff
                                    
        #plt.set_cmap('hot')
        f, axarr = plt.subplots(3)
        im = axarr[0].imshow(heat_map) #+  img/2)        
        im = axarr[1].imshow(img)
        im = axarr[2].imshow(sample_map)
        plt.show()



def create_mean(data, filename):
    mu = np.mean(data, axis = 0)
    with open(filename, "wb") as f:
        pickle.dump(mu, f)            
def load_mean(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)            


def load_test_images():    
    test_pattern = "{0}/test*small.png".format(PatDataDirectory)
    result = []
    for p in glob.glob(test_pattern):
        data = matplotlib.image.imread(p)
        result.append(data)
    return result            


patchSize = 64
samplesPerImage = 200
#generate_data(patchSize, samplesPerImage)
create_data(patchSize, samplesPerImage)

modelFilename = "./SavedModels/stroma_model.pkl"
muFilename = "./SavedModels/stroma_mu.pkl"
trainX, trainY, validationX, validationY = get_data_sets()
create_mean(trainX, muFilename)    
mu =load_mean(muFilename)


def t():
    trainX, trainY, validationX, validationY = get_data_sets()
    train(trainX, trainY, validationX, validationY, modelFilename, patchSize, mu)

#t()

def v():
    test_images = load_test_images()
    test(modelFilename, test_images, patchSize, mu)
v()