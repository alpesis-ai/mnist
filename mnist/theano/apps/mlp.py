import sys
import timeit

import numpy as np

import theano
import theano.tensor as T

from models.mlp import MLP
from datasets.data import load_data


def classifier_mlp(learning_rate=0.01,
                   l1_reg=0.00,
                   l2_reg=0.0001,
                   n_epochs=1000,
                   dataset='../../data/mnist.pkl.gz',
                   batch_size=20,
                   n_hidden=500):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient).

    :type l1_reg: float
    :param l1_reg: L1-norm's weight when added to the cost (see
                   regularization)

    :type l2_reg: float
    :param l2_reg: L2-norm's weight when added to the cost (see
                   regularization)

    :type n_epochs: string
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                    http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
    """

    datasets = load_data(dataset)
    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size

    # build the model
    print("... building the model")

    # allocate symbolic variables for the data
    # index to a minibatch
    index = T.lscalar() 
    # the data is presented as rasterized images
    x = T.matrix('x')
    # the labels are presented as 1D vector of int labels
    y = T.ivector('y')

    rng = np.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(rng=rng,
                     input=x,
                     n_in=28*28,
                     n_hidden=n_hidden,
                     n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2);
    # cost is expressed here symbolically
    cost = (classifier.negative_log_likelihood(y) +
            l1_reg * classifier.L1 +
            l2_reg * classifier.L2_sqr)

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    givens = {
        x: test_x[index * batch_size : (index+1) * batch_size],
        y: test_y[index * batch_size : (index+1) * batch_size]
    }
    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens=givens)

    givens = {
        x: valid_x[index * batch_size : (index+1) * batch_size],
        y: valid_y[index * batch_size : (index+1) * batch_size]
    }
    valid_model = theano.function(inputs=[index],
                                  outputs=classifier.errors(y),
                                  givens=givens)

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameteres of the model as a list of
    # (variable, update expression) pairs
    
    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size,
    # where each element is a pair formed from the two lists:
    # C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    givens = {
        x: train_x[index * batch_size : (index+1) * batch_size],
        y: train_y[index * batch_size : (index+1) * batch_size]
    }
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens=givens)


    # train model
    print('... training')

    # early-stopping parameters
    patience = 1000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [valid_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)
 
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    print(('epoch %i, minibatch %i/%i, test error of best model %f %%')
                          % (epoch, minibatch_index+1, n_train_batches, test_score*100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with best performance %f %%') %
           (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    # print(('The code for file ' +
    #       os.path.split(__file__)[1] +
    #       ' ran for %.2f' % ((end_time - start_time) / 60.)), file=sys.stderr)
