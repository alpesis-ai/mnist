import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from mlp import HiddenLayer
from denoising_autoencoder import DenoisingAutoEncoder
from logistic_regression import LogisticRegression


class StackedDenoisingAutoEncoder(object):
    """ Stacked Denoising Auto-Encoder

    A stacked denoising autoencoder is obtained by stacking several
    denoising autoencoder.

    The hidden layer of the denoising autoencoder at layer `i` becomes
    the input of the layer `i+1`.
    """

    def __init__(self, numpy_rng,
                       theano_rng=None,
                       n_ins=784,
                       hidden_layers_sizes=[500, 500],
                       n_outs=10,
                       corruption_levels=[0.1, 0.1]):

        """
        This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw
                          initial weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given, one
                           is generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                                    at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each layer
        """

        self.sigmoid_layers = []
        self.da_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2**30))

        # allocate symbolic variables for the data
        # the data is presented as rasterized images
        self.x = T.matrix('x')
        # the labels are presented as 1D vector of int labels
        self.y = T.ivector('y')

        # SDA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders.
        # We will first construct the SDA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SDA by doing
        # stochastich gradient descent on the MLP

        for i in range(self.n_layers):
            # construct the sigmoidal layer
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i-1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SDA if you are on the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question ...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDA
            # the visible biases in the DA are parameters of those DA
            # but not the SDA
            self.params.extend(sigmoid_layer.params)

            # construct a denoising autoencoder that shared weights with this
            # layer
            da_layer = DenoisingAutoEncoder(numpy_rng=numpy_rng,
                                            theano_rng=theano_rng,
                                            input=layer_input,
                                            n_visible=input_size,
                                            n_hidden=hidden_layers_sizes[i],
                                            W=sigmoid_layer.W,
                                            bhid=sigmoid_layer.b)
            self.da_layers.append(da_layer)

        # we now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(input=self.sigmoid_layers[-1].output,
                                           n_in=hidden_layers_sizes[-1],
                                           n_out=n_outs)

        self.params.extend(self.logLayer.params)

        # construct a function that implements one step of finetunining
        # compute the cost for second phase of training
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y.
        self.errors = self.logLayer.errors(self.y)


    def pretraining_functions(self, train_x, batch_size):
        """
        Generates a list of functions, each of them implementting one step
        in training the DA corresponding to the layer with same index.

        The function will require as input the minibatch index, and to train
        a DA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_x: theano.tensor.TensorType
        :param train_x: shared variable that contains all datapoints used
                        for training the DA

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the DA layers
        """

        # index to a minibatch
        index = T.lscalar('index')
        # % of corruption to use
        corruption_level = T.scalar('corruption')
        # learning rate to use
        learning_rate = T.scalar('lr')
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for da in self.da_layers:
            # get the cost and the updates list
            cost, updates = da.get_cost_updates(corruption_level,
                                                learning_rate)

            # compile the theano function
            fn = theano.function(inputs=[index,
                                         theano.In(corruption_level, value=0.2),
                                         theano.In(learning_rate, value=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x: train_x[batch_begin : batch_end]})

            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns


    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        """
        Generates a function `train` that implements one step of finetuning,
        a function `validate` that computes the error on a batch from the
        validation set, and a function `test` that computes the error on a 
        batch from the testing set.

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: it is a list that contain all the datasets;
                         the has to contain three paris, `train`, `valid`,
                         `test` in this order, where each pari is formed
                         of two theano variables, one for the datapoints,
                         the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch
  
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        """

        (train_x, train_y) = datasets[0]
        (valid_x, valid_y) = datasets[1]
        (test_x, test_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_x.get_values(borrow=True).shape[0]
        n_test_batches //= batch_size

        # index to a minibatch
        index = T.lscalar('index')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        givens = {
            self.x: train_x[index * batch_size : (index+1) * batch_size],
            self.y: train_y[index * batch_size : (index+1) * batch_size]
        }
        train_fn = theano.function(inputs=[index],
                                   outputs=self.finetune_cost,
                                   updates=updates,
                                   givens=givens,
                                   name='train')

        givens = {
            self.x: test_x[index * batch_size : (index+1) * batch_size],
            self.y: test_y[index * batch_size : (index+1) * batch_size]
        }
        test_score_i = theano.function([index],
                                       self.errors,
                                       givens=givens,
                                       name='test')

        givens = {
            self.x: valid_x[index * batch_size : (index+1) * batch_size],
            self.y: valid_y[index * batch_size : (index+1) * batch_size]
        }
        valid_score_i = theano.function([index],
                                        self.errors,
                                        givens=givens,
                                        name='valid')

        # create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score
