import numpy as np

import theano
import theano.tensor as T

from dataset.data import load_data
from models.conv import Conv


def classifier_lenet5(learning_rate=0.1,
                      n_epochs=200,
                      dataset='../../data/mnist.pkl.gz',
                      nkerns=[20, 50],
                      batch_size=500):
    """ Demonstrates lenet on MNIST dataset.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training/testing (MNIST)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = np.random.RandomState(23455)

    datasets = load_data(dataset)
    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_x.get_value(borrow=True).shape[0]
    n_test_batches = test_x.get_value(borrow=True).shape[0]

    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    # index to a minibatch
    index = T.lscalar()

    # the data is presented as rasterized images
    x = T.matrix('x')
    # the labels are presented as 1D vector of int labels
    y = T.ivector('y')

    # build the model
    print('... building the model')

    # reshape matrix of rasterized images of shape (batch_size, 28*28)
    # to a 4D tensor, compatible with our ConvLayer (28, 28)
    # is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 28, 28))

    # construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1, 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = Conv(rng,
                  input=layer0_input,
                  image_shape=(batch_size, 1, 28, 28),
                  filter_shape=(nkerns[0], 1, 5, 5),
                  poolsize=(2, 2))

    # construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = Conv(rng,
                  input=layer0.output,
                  image_shape(batch_size, nkerns[0], 12, 12),
                  filter_shape=(nkerns[1], nkerns[0], 5, 5),
                  poolsize=(2,2))

    # the HiddenLayer being fully-connected, it operates on 2D matrics
    # of shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng,
                         input=layer2_input,
                         n_in=nkerns[1] * 4 * 4,
                         n_out=500,
                         activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output,
                                n_in=500,
                                n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    givens = {
        x: test_x[index * batch_size : (index+1) * batch_size],
        y: test_y[index * batch_size : (index+1) * batch_size]
    }
    test_model = theano.function([index],
                                 layer3.errors(y),
                                 givens=givens)

    givens = {
        x: valid_x[index * batch_size : (index+1) * batch_size],
        y: valid_y[index * batch_size : (index+1) * batch_size]
    }
    valid_model = theano.function([index],
                                  layer3.errors(y),
                                  givens=givens)

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter.
    # We thus create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    givens = {
        x: train_x[index * batch_size : (index+1) * batch_size],
        y: train_y[index * batch_size : (index+1) * batch_size]
    }
    train_model = theano.function([index],
                                  cost,
                                  updates=updates,
                                  givens=givens)


    # train the model
    print('...training')
    # early-stopping parameters
    # look as this many examples regardless
    patience = 10000
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    # go through this many minibatche before checking the network
    # on the validation set, in this case we check every epoch
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
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if iter % 100 == 0:
                print("training @ iter = ", iter)
            cost_ij = train_model(minibatch_index)

            if (iter+1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch,
                       minibatch_index + 1,
                       n_train_batches,
                       this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    # improve patience if loss imporovement is good enough
                    if this_validation_loss < best_validation_loss * imporovement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(("epoch %i, minibatch %i/%i, test error of best model %f %%") %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print("Optimization complete.")
    print("Best validation score of %f %% obtained at iteration %i, "
          "with test performance %f %%" %
          (best_validation_loss * 100.,
           best_iter + 1,
           test_score * 100.))

    print("The code for file " +
          os.path.split(__file__)[1] +
          " ran for %.2fm" % ((end_time - start_time) / 60.)), file=sys.stderr)


def experiment(state, channel):
    classifier_lenet5(state.learning_rate, dataset=state.dataset)
