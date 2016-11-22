import numpy as np

import theano
import theano.tensor as T


class LogisticRegression(object):
    """ Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`.

    Classification is done by projecting data points onto a set of
    hyperplanes, the distance to which is used to determine a class
    membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression.
        
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie
        """

        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                              dtype=theano.config.floatX),
                               name='W',
                               borrow=True)
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                               dtype=theano.config.floatX),
                               name='b',
                               borrow=True)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # where:
        # - W: a matrix where column-k represent the separation hyperplane
        #      for class-k
        # - x: a matrix where row-j represents input training sample-j
        # - b: a vector where element-k represents the free parameter of
        #      hyperplane-k
        # - softmax: normalization
        self.y_hat = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        # returns the maximum values along an axis
        self.y_pred = T.argmax(self.y_hat, axis=1)


        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


    def negative_log_likelihood(self, y):
        """ Returns the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that the learning rate
              is less dependent on the batch size.
        """

        return -T.mean(T.log(self.y_hat)[T.arange(y.shape[0]), y])


    def errors(self, y):
        """ Returns a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch; zero one loss over
        the size of the minibatch.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the sampe shape as self.y_pred",
                            ('y', y.type, 'y_pred', self.y_pred.type))

        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
