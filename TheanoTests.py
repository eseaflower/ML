import time
import numpy
import theano
import theano.tensor as T
import MNISTUtil
from Layers import LogisticRegression
from Layers import nnlayer
from Trainer import Trainer

rng = numpy.random

#theano.config.profile = True


l = MNISTUtil.MNISTLoader()
train_lables_filename = r"../Data/MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
train_data_filename = r"../Data/MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte"
train_set_x, train_set_y, test_set_x, test_set_y = l.Load(train_data_filename, train_lables_filename, 0.9)



# allocate symbolic variables for the data
x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

rng = numpy.random.RandomState(1234)

# the cost we minimize during training is the negative log likelihood of
# the model in symbolic format
classifier = nnlayer.MLP(rng=rng, input=x, topology=[28*28, 500, 10], rectified=True, dropout_rate=0.4)
cost = classifier.negative_log_likelihood(y) + 0.0001*classifier.L2_sqr
costFunction = (classifier.params, cost)


# Create trainer
tt = Trainer.MLPBatchTrainer()
tt.train(x, y, costFunction, train_set_x, train_set_y, epochs=15,learning_rate=0.001, rms=True)


batch_size=1
index = T.lscalar()  # index to a [mini]batch
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
print("Errors " , errs)
