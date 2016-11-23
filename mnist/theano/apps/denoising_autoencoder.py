from __future__ import print_function

import sys
import timeit
import os.path

import numpy as np
try:
    import PIL.Image as Image
except ImportError:
    import Image


import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from datasets.data import load_data
from utils.utils import tile_raster_images
from models.denoising_autoencoder import DenoisingAutoEncoder


def classifier_denoising_autoencoder(learning_rate=0.1,
                                     training_epochs=15,
                                     dataset='../../data/mnist.pkl.gz',
                                     batch_size=20,
                                     output_folder='outputs/DA'):
    """
    :type learning_rate: float
    :param learning_rate: learning rate used for training the Denoising
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset
    """

    datasets = load_data(dataset)
    train_x, train_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    # index to a minibatch
    index = T.lscalar()
    # the data is presented as rasterized images
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)


    # build the model no corruption
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = DenoisingAutoEncoder(numpy_rng=rng,
                              theano_rng=theano_rng,
                              input=x,
                              n_visible=28*28,
                              n_hidden=500)

    cost, updates = da.get_cost_updates(corruption_level=0.,
                                        learning_rate=learning_rate)

    givens = {
        x: train_x[index * batch_size : (index+1) * batch_size]
    }
    train_da = theano.function([index],
                               cost,
                               updates=updates,
                               givens=givens) 

    start_time = timeit.default_timer()

    # train
    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, np.mean(c, dtype='float64')) 

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print(('The no corruption code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((training_time) / 60.)), file=sys.stderr)

    image = Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T,
                                               img_shape=(28, 28),
                                               tile_shape=(10, 10),
                                               tile_spacing=(1, 1)))
    image.save('../outputs/filters_corruption_0.png')


    # building the model corruption 30%
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2**30))

    da = DenoisingAutoEncoder(numpy_rng=rng,
                              theano_rng=theano_rng,
                              input=x,
                              n_visible=28*28,
                              n_hidden=500)

    cost, updates = da.get_cost_updates(corruption_level=0.3,
                                        learning_rate=learning_rate)

    givens = {
        x: train_x[index * batch_size : (index+1) * batch_size]
    }
    train_da = theano.function([index],
                               cost,
                               updates=updates,
                               givens=givens)

    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        c = []
        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))

        print('Training epoch %d, cost ' % epoch, np.mean(c, dtype='float64'))

    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print(("The 30% corruption code for file " +
           os.path.split(__file__)[1] +
           " ran for %.2fm" % (training_time / 60.)), file=sys.stderr)

    image = Image.formarray(tile_raster_images(X=da.W.get_value(borrow=True).T,
                                               img_shape=(28, 28),
                                               tile_shape=(10, 10),
                                               tile_spacing=(1, 1)))
    image.save('../outputs/filters_corruption_30.png')
   
    # os.chdir('../') 
