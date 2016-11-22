""" LeNet5 Neural Network Architecture

LeNet5 is a convoluational neural network, good for classifying images.

This implementation simplifies the model in the following ways:

- LeNet doesn't implement location-specific gain and bias parameters
- LeNet doesn't implement pooling by average, it implements pooling by max
- Digit classification is implemented with a logistic regression rather
  than an RBF network
- LeNet5 was not fully-connected convolutions at second layer

References:
    Gradient-based Learning Applied to Document
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
"""

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool


class Conv(object):

    def __init__(self, rng,
                       input,
                       filter_shape,
                       image_shape,
                       poolsize=(2,2)):
        """
        Allocate a LeNet with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
       
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]

        self.input = input

        # num of inputs (each hidden unit) =
        # num_input_feature_maps * filter_height * filter_width
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # fan_out = num_output_feature_maps * filter_height * filter_width
        #            / pooling_size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) //
                  np.prod(poolsize)

        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound,
                                                      high=W_bound,
                                                      size=filter_shape),
                                          dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -> one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(input=input,
                          filters=self.W,
                          filter_shape=filter_shape,
                          input_shape=image_shape)

        # pool each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(input=conv_out,
                                  ds=poolsize,
                                  ignore_border=True)

        # add the bias term
        # Since the bias is a vector (1D array), we first reshape
        # it to a tensor of shape (1, n_filters, 1, 1).
        # Each bias will thus be broadcasted across mini-batches
        # and feature mamp width and height.
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
