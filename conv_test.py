import time
import numpy
import theano
import theano.tensor as T
import MNISTUtil
from Layers import LogisticRegression
from Layers import nnlayer
from Trainer import Trainer
from Layers import ConvNet
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import plotutils

theano.config.allow_gc = False

rng = numpy.random.RandomState(23455)

l = MNISTUtil.MNISTLoader()
train_lables_filename = "./train-labels-idx1-ubyte/train-labels.idx1-ubyte"
train_data_filename = "./train-images-idx3-ubyte/train-images.idx3-ubyte"
train_set_x, train_set_y, test_set_x, test_set_y = l.Load(train_data_filename, train_lables_filename, 0.2)


batch_size = 256

nkerns=[20, 50]

# allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                    # [int] labels

######################
# BUILD ACTUAL MODEL #
######################
print('... building the model')

# Reshape matrix of rasterized images of shape (batch_size,28*28)
# to a 4D tensor, compatible with our LeNetConvPoolLayer
layer0_input = x.reshape((batch_size, 1, 28, 28))

# Construct the first convolutional pooling layer:
# filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
# maxpooling reduces this further to (24/2,24/2) = (12,12)
# 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
layer0 = ConvNet.LeNetConvPoolLayer(rng, input=layer0_input,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))


# Construct the second convolutional pooling layer
# filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
# maxpooling reduces this further to (8/2,8/2) = (4,4)
# 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
layer1 = ConvNet.LeNetConvPoolLayer(rng, input=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

# the HiddenLayer being fully-connected, it operates on 2D matrices of
# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
# This will generate a matrix of shape (20,32*4*4) = (20,512)
layer2_input = layer1.output.flatten(2)

tst = nnlayer.ConvNet(rng, x, (256, 28, 28), [(20, 5, 5), (50, 5, 5)], rectified=False)

#classifier = nnlayer.MLP(rng, layer2_input, [nkerns[1] * 4 * 4, 256, 10])
classifier = nnlayer.MLP(rng, tst.output, [tst.output_size, 256, 10])
cost = classifier.negative_log_likelihood(y)
#params = layer0.params + layer1.params + classifier.params
params = tst.params + classifier.params
costFunction = (params, cost)

tt = Trainer.MLPBatchTrainer()
tt.train(x, y,costFunction, train_set_x, train_set_y, batch_size, epochs=3, rms=False)

#ten = layer0.W.get_value(borrow=True)
#plotutils.plot_tensor(ten)


test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

n_test_batch = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)
test_loss = [test_model(i) for i in range(n_test_batch)]
errs = numpy.mean(test_loss);
#print "Test losses: ", test_loss
#errs = test_model(0)
print("Errors ", errs)


