import time
import numpy
import theano
import theano.tensor as T
import MNISTUtil
from Layers import LogisticRegression
from Layers import nnlayer
from Trainer.Trainer import MLPBatchTrainer, VariableAndData
import matplotlib.pyplot as plt
import plotutils

rng = numpy.random

#theano.config.profile = True


l = MNISTUtil.MNISTLoader()
train_lables_filename = r"../Data/MNIST/train-labels-idx1-ubyte/train-labels.idx1-ubyte"
train_data_filename = r"../Data/MNIST/train-images-idx3-ubyte/train-images.idx3-ubyte"
train_set_x, train_set_y, validation_set_x, validation_set_y = l.Load(train_data_filename, train_lables_filename, 0.9)



# allocate symbolic variables for the data
x = T.matrix('x')  # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

rng = numpy.random.RandomState(1234)

# Use convnet
#conv_net = nnlayer.ConvNet(rng, x, (28, 28), [(20, 5, 5), (50, 5, 5)], rectified=True)
conv_net = nnlayer.ConvNet(rng, x, (28, 28), [(20, 5, 5)], rectified=True)

# the cost we minimize during training is the negative log likelihood of
# the model in symbolic format
classifier = nnlayer.MLPReg(rng=rng, input=conv_net.output, topology=[(conv_net.output_size,), 
                                          #(256, nnlayer.TanhLayer), 
                                          (10, nnlayer.LogisticRegressionLayer)])
cost = classifier.cost(y) + 0.0001*classifier.L2_sqr + 0.0001*conv_net.L2_sqr
params = conv_net.params + classifier.params
costFunction = (params, cost)

# Create trainer
tt = MLPBatchTrainer()
valid_func = theano.function(inputs = [],
                        outputs = [classifier.cost(y)],
                        givens = {x:validation_set_x, y:validation_set_y})                            

variableAndData = (VariableAndData(x, train_set_x), VariableAndData(y, train_set_y, size=train_set_x.get_value(borrow=True).shape[0]))
epochFunction, stateMananger = tt.getEpochTrainer(costFunction, variableAndData, batch_size=64, rms = True, momentum=0.9) 
    
# Train with adaptive learning rate.
stats = tt.trainALR(epochFunction, 
                    valid_func, 
                    initial_learning_rate=0.001, 
                    epochs=3, 
                    convergence_criteria=0.0001, 
                    max_runs=100,
                    state_manager = stateMananger)

validation_scores = [item["validation_score"] for item in stats]
train_scorees = [item["training_costs"][-1] for item in stats]
#train_scorees = stats[0]["training_costs"]
plt.plot(validation_scores, 'g')
plt.plot(train_scorees, 'r')
plt.show()

# Plot the learned filters.
plotutils.plot_tensor_image(conv_net.layers[0].W.get_value())


test_lables_filename = r"../Data/MNIST/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
test_data_filename = r"../Data/MNIST/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
test_set_x, test_set_y, dummy_x, dummy_y = l.Load(test_data_filename, test_lables_filename)

test_func = theano.function(inputs = [],
                        outputs = [classifier.errors(y)],
                        givens = {x:test_set_x, y:test_set_y})                            

avg_error = test_func()[0]

print("Average error: {0}".format(avg_error))