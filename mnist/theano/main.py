import gzip
import cPickle

import theano
import numpy as np


# size of the minibatch
BATCH_SIZE = 500


from data import load_data


def shared_dataset(data_xy):
    """
    Function that loads the dataset into shared variables.

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory).
    """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))

    return shared_x, theano.tensor.cast(shared_y, 'int32')


def sgd_optimization_mnist(learning_rate=0.13,
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
    
    dataset = load_data(dataset)


if __name__ == '__main__':

    # f = gzip.open('../../data/mnist.pkl.gz', 'rb')
    # train, valid, test = cPickle.load(f)
    # f.close()

    # train_x, train_y = shared_dataset(train)
    # valid_x, valid_y = shared_dataset(valid)
    # test_x, test_y = shared_dataset(test)

    # data = train_x[2*BATCH_SIZE : 3*BATCH_SIZE]
    # label = train_y[2*BATCH_SIZE : 3*BATCH_SIZE]

    sgd_optimization_mnist()
