from __future__ import print_function
import sys
import gzip
import timeit
import os.path

import numpy as np
import six.moves.cPickle as pickle

import theano
import theano.tensor as T


from models.logistic_regression import LogisticRegression


# size of the minibatch
BATCH_SIZE = 500


from datasets.data import load_data


def sgd_optimization(learning_rate=0.13,
                     n_epochs=1000,
                     dataset='../../data/mnist.pkl.gz',
                     batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: int
    :param dataset: the path of MNIST dataset file from 
                    http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    :type batch_size: int
    :param batch_size: 
    """
    
    datasets = load_data(dataset)
    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]


    # build the model
    print("... building the model")
    # allocate symbolic variables for the data
    # index to a minibatch
    index = T.lscalar()

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # construct the logistic regression class
    # each MNIST image has size 28*28
    classifier = LogisticRegression(input=x,
                                    n_in=28*28,
                                    n_out=10)
    # the cost we minimize during training is the negative log likelihood
    # of the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_givens = {
        x: test_x[index * batch_size : (index+1) * batch_size],
        y: test_y[index * batch_size : (index+1) * batch_size]
    }
    test_model = theano.function(inputs=[index],
                                 outputs=classifier.errors(y),
                                 givens=test_givens)
    valid_givens = {
        x: valid_x[index * batch_size : (index+1) * batch_size],
        y: valid_y[index * batch_size : (index+1) * batch_size]
    }
    valid_model = theano.function(inputs=[index],
                                  outputs=classifier.errors(y),
                                  givens=valid_givens)

    # compute the gradient of cost with respect to theta = (W, b)
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [
        (classifier.W, classifier.W - learning_rate * g_W),
        (classifier.b, classifier.b - learning_rate * g_b)
    ]
    
    # compiling a Theano function train_model that returns the cost,
    # but in the same time updates the parameter of the model based on
    # the rules defined in `updates`.
    train_givens = {
        x: train_x[index * batch_size : (index+1) * batch_size],
        y: train_y[index * batch_size : (index+1) * batch_size]
    }
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens=train_givens)


    # train model
    print('... training the model')
    # early-stopping parameters
    # look as this many examples regardless
    patience = 5000
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    # go through this many
    # minibatch before checking the network
    # on the validation set
    # in this case we check every epoch
    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = np.inf
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
                    "epoch %i, minibatch %i/%i, validatioin error %f %%" %
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
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    
                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(
                        ('epoch %i, minibatch %i/%i, test error of best model %f %%') %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('outputs/best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        ('Optimization complete with best validation score of %f %%, '
         'with best performance %f %%') %
        (best_validation_loss % 100., test_score * 100.)
    )
    print('The code run for %d epochs, with %f epochs/sec' %
          (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)


def predict():
    """
    An example of how to load a trained model and use it to
    predict labels.
    """

    # load the saved model
    classifier = pickle.load(open('outputs/best_model.pkl'))

    # compile a predictor function
    predict_model = theano.function(inputs=[classifier.input],
                                    outputs=classifier.y_pred)

    # we can test it on some examples from test case
    dataset = '../../data/mnist.pkl.gz'
    datasets = load_data(dataset)
    test_x, teste_y = datasets[2]
    test_x = test_x.get_value()

    predicted_values = predict_model(test_x[:10])
    print("predicted values for the first 10 examples in teste set:")
    print(predicted_values)
