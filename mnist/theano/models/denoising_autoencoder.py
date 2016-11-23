""" Denoising AutoEncoder

Denosing autoencoders are the building blocks for SDA.
They are based on auto-encoders as the ones used in Bengio et al. 2007.
"""

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class DenoisingAutoEncoder(object):
    """ Denoising Auto-Encoder

    A denoising autoencoders tries to reconstruct the input from a
    corrupted version of it by projecting it first in a latent space and
    reprojecting it afterwards back in the input space.
    """

    def __init__(self, numpy_rng,
                       theano_rng=None,
                       input=None,
                       n_visible=784,
                       n_hidden=500,
                       W=None,
                       bhid=None,
                       bvis=None):
        """
        Initialize the denoising autoencoder class by specifying:
        - the number of visible units (the dimension of the input)
        - the number of hidden units (the dimension of the latent/hidden space
        - the corruption level

        The constructor also receives symbolic variables for the inputs,
        weights and bias.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator, if None is given, one
                           is generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone denoising auto-encoder

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should
                  be shared belong the denoising auto-encoder and another
                  architecture. If denoising auto-encoder should be
                  standalone, set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values
                     (for hidden units) that should be shared belong
                     denoising auto-encoder and another architecture;
                     if denoising auto-encoder should be standalone,
                     set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values
                     (for visible units) that should be shared belong
                     denoising auto-encoder and another architecture;
                     if denoising auto-encoder should be standalone, set
                     this to None
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note: W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W: is initialized with `initial_W` which is uniformely sampled
            #    from -4*sqrt(6./(n_visible+n_hidden)) and
            #          4*sqrt(6./(n_visible+n_hidden))
            initial_W = np.asarray(
                numpy_rng.uniform(low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                                  high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                                  size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX)

            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=np.zeros(n_visible, dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transponse
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]


    def get_corrupted_input(self, input, corruption_level):
        """
        This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
    
        :type input: theano.rng.binomial
        :param input: the shape (size) of random numbers that it should produce

        :type corruption_level: 
        :param corruption_level: the number of trials
        """

        return self.theano_rng.binomial(size=input.shape,
                                        n=1,
                                        p=1-corruption_level,
                                        dtype=theano.config.floatX) * input


    def get_hidden_values(self, input):
        """
        Computes the values of the hidden layer
        """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)


    def get_reconstructed_input(self, hidden):
        """
        Computes the reconstructed input given the values of the hidden
        layer.
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)


    def get_cost_updates(self, corruption_level, learning_rate):
        """
        This function computes the cost and the updates for one training
        step of the denosing auto-encoder.
        """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        # note: we sum over the size of a datapoint
        #       if we are using minibatches, L will be a vector,
        #       with one entry per example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        # note: L is now a vector, where each element is the cross-entropy
        #       cost of the reconstruction of the corresponding example of
        #       the minibatch. 
        #       We need to compute the average of all these to get the cost
        #       of the minibatch.
        cost = T.mean(L)

        # compuate the gradients of the cost of the denoising auto-encoder
        # with respect to its parameters
        gparams = T.grad(cost, self.params)

        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        
        return (cost, updates)
   
