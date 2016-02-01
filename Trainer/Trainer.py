import numpy
import theano
import theano.tensor as T
import time
import math
import matplotlib.pyplot as plt
from lasagne.updates import rmsprop
import pickle
import os


from msvcrt import kbhit, getch


class VariableAndData(object):
    def __init__(self, variable, data, size = None):
        self.variable = variable
        self.data = data
        if not size:
            #If size is not given assume it is a shared variable
            size = self.data.get_value(borrow=True).shape[0]
        self.dataSize = size
    def slice(self, start, end):
        return self.data[start:end]

class StateManager(object):
    def __init__(self, params, resetActions = None):
        self.params = params
        self.resetActions = resetActions
        self.storedState = None        
        self.initStat(r".\training_stats.pkl")        


    def restore(self):
        if self.storedState:
            for paramIndex in range(len(self.params)):
                param = self.params[paramIndex]
                value = self.storedState[paramIndex]
                param.set_value(value)
    
    def reset(self):
        if self.resetActions:
            for arg, action in self.resetActions:
                action(arg)

    def save(self):
        # Get the current value of the parameters and store them
        self.storedState = []
        for param in self.params:
            # Assume they are all theano.shared
            self.storedState.append(param.get_value())

    def initStat(self, filename):
        self.statFile = filename
        if os.path.isfile(self.statFile):
            os.remove(self.statFile)
        self.hasFile = False

    def addStatistics(self, stats):
        existing_data = []
        if self.hasFile:
            with open(self.statFile, "rb") as f:
                existing_data = pickle.load(f)        
        existing_data.append(stats)
        with open(self.statFile, "wb") as f:
            pickle.dump(existing_data, f)
        self.hasFile = True

class MLPBatchTrainer(object):
    def __init__(self):
        return
    
    def getUpdates(self, cost, params, lr, rms=True, momentum=None):

        #if rms:
        # odict = rmsprop(cost, params, lr)
        #    u = []
        #    for k in odict:
        #        u.append((k, odict[k]))

        #    return u, []

        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)
                            
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        rho = 0.9
        eps = 1e-6
        updates = []
        resetActions = []
        if rms:
            # Use RMSProp (scale the gradient).
            for param, gparam in zip(params, gparams):
                acc = theano.shared(param.get_value() * 0.)                                
                acc_new  = rho*acc + (1.0-rho)*gparam**2
                updates.append((acc, acc_new)) 
                g_scale = T.sqrt(acc_new + eps)
                gparam = gparam/g_scale
                # gparam is the gradient step.
                step = -lr*gparam
                if momentum:
                    paramValue = param.get_value()
                    v = theano.shared(paramValue*0.)
                    v_new = momentum*v + step
                    updates.append((v, v_new))                    
                    resetActions.append((v, lambda p: p.set_value(p.get_value()*0.)))
                    step = momentum*v_new - step

                updates.append((param, param + step)) 

        else:                    
            # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
            # same length, zip generates a list C of same size, where each element
            # is a pair formed from the two lists :
            #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
            for param, gparam in zip(params, gparams):
                updates.append((param, param - lr * gparam)) 
        
        return updates, resetActions

    @staticmethod
    def wrapUpdate(update):        
        def inner(cost, params, lr):
            resetActions = []
            updates = []
            updateDict = update(cost, params, lr)
            for key in updateDict:
                updates.append((key, updateDict[key]))
            return updates, resetActions
        return inner

    def getMinibatchTrainer(self, costFunction, variableToData, rms=True, momentum = None, updateFunction = None):
        # define params        
        lr = T.fscalar('lr')    
        start = T.iscalar('start')
        end = T.iscalar('end')

        # Get the cost and its parameters.
        params = costFunction[0]
        cost = costFunction[1]

        if updateFunction:
            updates, resetActions = updateFunction(cost, params, lr)
        else:
            # Get the updates.
            updates, resetActions = self.getUpdates(cost, params, lr, rms, momentum)
        # Store all state variables.
        stateManager = StateManager([u[0] for u in updates], resetActions)

        # Slice the data
        givens = dict()
        for item in variableToData:
            givens[item.variable] = item.slice(start,end)
        
                
        # Define the training function.
        train_model = theano.function(inputs=[theano.Param(start, borrow = True), 
                                              theano.Param(end, borrow=True), 
                                              theano.Param(lr, borrow=True)],
                                        outputs=theano.Out(cost, borrow=True),
                                        updates=updates,
                                        givens=givens)

        
        return train_model, stateManager

    def getEpochTrainer(self, costFunction, variableToData, batch_size = 512, rms=True, momentum = None, randomize=False, updateFunction = None):        
        miniBatchFunction, stateManager = self.getMinibatchTrainer(costFunction, variableToData, rms, momentum, updateFunction)        
        n_samples = variableToData[0].dataSize
        batch_size = min(batch_size, n_samples)
        n_batches =  math.ceil(n_samples / batch_size)

        # Use a deterministic state
        rs = numpy.random.RandomState(1234)                                                
        
        # Define function that runs one epoch
        def runEpoch(lr):                
            accumulated_cost = 0.
            used_range = range(n_batches)
            if randomize:                
                used_range = rs.permutation(used_range)
            #c = []
            for index in used_range:
                start_index = index * batch_size
                end_index = min((index + 1) * batch_size, n_samples)
                minibatchCost = miniBatchFunction(start_index, end_index, lr)                
                #c.append(minibatchCost)
                # Add cost relative to the batch size.
                accumulated_cost += minibatchCost * ((end_index - start_index) / batch_size)                    
            
            #plt.plot(c)    
            #plt.show()                
            return accumulated_cost / n_batches
        return runEpoch, stateManager

    def train(self, x, y, costFunction, train_set_x, train_set_y, batch_size=512, learning_rate=0.1, epochs=10, max_batches = -1, weight=None, train_weight = None, rms=True):
        """
        classifier: The classifier to train.
        x: Symbolic input
        y: Symbolic labels
        costFunction: tuple, (params, expression of cost to optimize)
        train_set_x: Input training set
        train_set_y: Input training labels
        """       

        # Get the cost and its parameters.
        params = costFunction[0]
        cost = costFunction[1]

        # compute the gradient of cost with respect to theta (sotred in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        rho = 0.9
        eps = 1e-6
        updates = []
        if rms:
            # Use RMSProp (scale the gradient).
            for param, gparam in zip(params, gparams):
                acc = theano.shared(param.get_value() * 0.)
                acc_new  = rho*acc + (1.0-rho)*gparam**2
                g_scale = T.sqrt(acc_new + eps)
                gparam = gparam/g_scale
                updates.append((acc, acc_new)) 
                updates.append((param, param - learning_rate * gparam)) 
        else:                    
            # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
            # same length, zip generates a list C of same size, where each element
            # is a pair formed from the two lists :
            #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
            for param, gparam in zip(params, gparams):
                updates.append((param, param - learning_rate * gparam)) 
                


        # Compute actual batch size
        n_samples = train_set_x.get_value(borrow=True).shape[0]
        batch_size = min(batch_size, n_samples)

        index = T.lscalar()  # index to a [mini]batch
        if not weight:
            # Define the training function.
            train_model = theano.function(inputs=[theano.Param(index, borrow=True)],
                                          outputs=theano.Out(cost, borrow=True),
                                          updates=updates,
                                          givens={
                                               x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                               y: train_set_y[index * batch_size:(index + 1) * batch_size]})
        else:
            # Define the training function.
            train_model = theano.function(inputs=[index],
                                          outputs=cost,
                                          updates=updates,
                                          givens={
                                               x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                               y: train_set_y[index * batch_size:(index + 1) * batch_size],
                                               weight: train_weight[index * batch_size:(index + 1) * batch_size]})

        # Compute number of batches to run.
        n_batches = int(n_samples / batch_size)
        if max_batches != -1:
            n_batches = min(max_batches, n_batches)
        
            
        #print(train_model.maker.fgraph.toposort())
        #input('Press smthn')

        print("Batches: ", n_batches)
               
        prev_cost= 0.
        result = []
        for e in range(epochs):
            print("Startng epoch ", e)
            avg_cost = 0.
            for i in range(n_batches):
                c = train_model(i)                
                avg_cost += c
                #print "Cost: ", c
            avg_cost /= n_batches
            print("Average cost {0}, Cost diff {1}".format(avg_cost, prev_cost - avg_cost))
            prev_cost = avg_cost
            result.append(avg_cost)

        return result


    def trainALR(self, train_function, validation_function, 
                 initial_learning_rate=0.001, epochs=100, 
                 convergence_criteria=0.001,
                 max_runs = 30,
                 state_manager = None):            
        """
           train_function - Function that has the current learning rate as parameter.
           validation_function - Function that returns a validation score.
           initial_learning_rate - The initial guess for the learning rate.
           epochs - The number of epochs to run before each validation.
           convergence_criteria - If the validation score is below this value the function returns.
           max_runs - The maximum number of runs to make.
        """

        best_score = None        
        current_learning_rate = initial_learning_rate
        
        # Keep track of when we started.
        t0 = time.time()

        learning_rate_eps = 1e-8
        max_learning_rate = 0.9
        min_learning_rate = 1e-9

        training_statistics = []
        improve_eps = 1e-6
        improve_fraction = 1e-4        
        patience = 5
        remaining_patience = patience

        for i in range(max_runs):
            
            # Do the first training
            epoch_costs = [train_function(current_learning_rate) for i in range(epochs)]
            
            print("Last training cost: {0}".format(epoch_costs[-1]))


            validation_score = validation_function()[0]                             

            if validation_score < convergence_criteria:
                print("Converged")
                break

            if not best_score:
                # Update the previous to get a baseline
                print("First iteration. Baseline: {0}".format(validation_score))
                best_score = validation_score
                if state_manager:
                    print("Saving baseline parameters")
                    state_manager.save()
                    print("Done writing state")
                continue

            # Compute score delta
            score_delta = best_score - validation_score
            relative_score_delta = score_delta / (current_learning_rate*best_score)
            iteration_statistics = {"training_costs":epoch_costs, 
                                    "validation_score":validation_score, 
                                    "learning_rate": current_learning_rate,
                                    "score_delta":score_delta}
            
            delta_fraction = score_delta / best_score
            print("{0} of {1} - Score: {2}, Delta: {3}".format(i, max_runs, validation_score, score_delta))

            training_statistics.append(iteration_statistics)
                                            
            # Update learning rate based on relative validation score delta.
            previous_learning_rate = current_learning_rate                        


            if score_delta > 0:
                # Set previous to current
                best_score = validation_score
                remaining_patience = patience
                if state_manager:
                    # Save a new best state
                    print("Overwriting state with new best values.")
                    state_manager.save()                                        
            #elif score_delta < 0:
            elif delta_fraction < improve_fraction:                
                print("Delta fraction: {0} below {1}".format(delta_fraction, improve_fraction))
                remaining_patience -= 1
                if remaining_patience < 0:
                    current_learning_rate *= 0.3                
                    remaining_patience = patience                
                    if state_manager:
                        print("Restoring state to previous best values")
                        state_manager.restore()
                        state_manager.reset()
            
            # Cap the learning rate.
            current_learning_rate = float(numpy.max([numpy.min([current_learning_rate, max_learning_rate]), min_learning_rate]))        
        
            if current_learning_rate != previous_learning_rate:
                print("Learning rate {0} -> {1}".format(previous_learning_rate, current_learning_rate))

            # Exit if the learning rate is too small.
            if current_learning_rate < 1e-8:
                print("Learning rate vanished")
                break
                
        # Training time
        t1 = time.time()
        training_time = t1 - t0
        return training_statistics


    def trainALR2(self, train_function, validation_function, 
                 initial_learning_rate=0.001, epochs=100, 
                 convergence_criteria=0.001,
                 max_runs = 30,
                 state_manager = None):
        best_score = None        
        current_learning_rate = initial_learning_rate
        
        # Keep track of when we started.
        t0 = time.time()

        learning_rate_eps = 1e-8
        max_learning_rate = 0.9
        min_learning_rate = 1e-9

        training_statistics = []
        improve_eps = 1e-6
        improve_fraction = 1e-4        
        patience = 5
        remaining_patience = patience

        ts = CircularBuffer(10)

        for i in range(max_runs):
            
            if kbhit():
                getch()
                cmd = input("Command (q, lr):")
                if cmd == "q":
                    print("Exiting...")
                    break
                elif cmd == "lr":
                    lr_str = input("New learning rate ({0}):".format(current_learning_rate))
                    current_learning_rate = float(lr_str)
                    ts.reset()


            # Do the first training
            epoch_costs = [float(train_function(current_learning_rate)) for i in range(epochs)]

            validation_score = validation_function()[0]                  
            print("{2}/{3} Training: {0}, Validation: {1}".format(epoch_costs[-1], validation_score, i, max_runs))           
            if validation_score < convergence_criteria:
                print("Converged")
                break

            if not best_score:
                # Update the previous to get a baseline
                print("First iteration. Baseline: {0}".format(validation_score))
                best_score = validation_score
                if state_manager:                    
                    state_manager.save()
                continue

            # Save training scores.
            for ec in epoch_costs:
                ts.push(ec)
            
            y_start = ts.first()
            y_end = ts.last()
            if y_start > 0 and y_end > 0:
                # We have a window of data.
                y_delta = y_start - y_end
                y_delta_fraction = y_delta / current_learning_rate
                print("Delta: {0}, Fraction: {1}".format(y_delta, y_delta_fraction))
                #if y_delta_fraction < improve_fraction:
                #    if y_delta_fraction < 0 and state_manager:                        
                #        print("Restoring previous best values")
                #        state_manager.restore()
                #        state_manager.reset()

                #    # Change learning rate.                            
                #    current_learning_rate *= 0.3
                #    ts.reset()

                #    # Exit if the learning rate is too small.
                #    if current_learning_rate < 1e-8:
                #        print("Learning rate vanished")
                #        break
                #    print("New learning rate: {0}".format(current_learning_rate))
                    


            # Compute score delta
            score_delta = best_score - validation_score
            iteration_statistics = {"training_costs":epoch_costs, 
                                    "validation_score":float(validation_score), 
                                    "learning_rate": current_learning_rate,
                                    "score_delta":score_delta}
            

            state_manager.addStatistics(iteration_statistics)

            training_statistics.append(iteration_statistics)
                                            
            if validation_score < best_score:
                # Set previous to current
                best_score = validation_score
                if state_manager:
                    # Save a new best state
                    print("Overwriting state with new best values.")
                    state_manager.save()                                        
            

                
        # Training time
        if state_manager:
            # Make sure we are at the best settings.
            state_manager.reset()
            state_manager.restore()

        t1 = time.time()
        training_time = t1 - t0
        return training_statistics


class CircularBuffer(object):
    def __init__(self, size):
        self.size = size
        self.reset()

    def push(self, sample):
        self.buffer[self.current_position] = sample
        self.current_position = (self.current_position + 1) % self.size

    def reset(self):
        self.current_position = 0
        self.buffer = [-1]*self.size

    def index(self, i):
        return (self.current_position +i) % self.size

    def item(self, i):
        return self.buffer[self.index(i)]

    def first(self):
        return self.item(0)
    def last(self):
        return self.item(-1)
