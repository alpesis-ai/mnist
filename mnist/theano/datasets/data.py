import gzip
import os.path

import numpy as np
import six.moves.cPickle as pickle

import theano
import theano.tensor as T


def load_data(dataset):
    """ Loads the dataset.

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """

    # load the dataset
    data_dir, data_file = os.path.split(dataset)

    if data_dir == "" and not os.path.isfile(dataset):
        raise IOError("File does not exist.")

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')
    with gzip.open(dataset, 'rb') as f:
        try:
            train, valid, test = pickle.load(f, encoding='latin1')
        except:
            train, valid, test = pickle.load(f)

    train_x, train_y = _shared_dataset(train)
    valid_x, valid_y = _shared_dataset(valid)
    test_x, test_y = _shared_dataset(test)
    rval = [(train_x, train_y),
            (valid_x, valid_y),
            (test_x, test_y)] 

    return rval


def _shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """

    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX),
                                        borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX),
                                        borrow=borrow)

    return shared_x, T.cast(shared_y, 'int32')
