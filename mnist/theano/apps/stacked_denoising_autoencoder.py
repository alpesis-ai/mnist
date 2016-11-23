from __future__ import print_function

import sys
import timeit
import os.path

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from datasets.data import load_data
from models.stacked_denoising_autoencoder import StackedDenoisingAutoEncoder


def classifier_sda(finetune_lr=0.1,
                   pretraining_epochs=15,
                   pretrain_lr=0.001,
                   training_epochs=1000,
                   dataset='../../data/mnist.pkl.gz',
                   batch_size=1):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
                          (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type training_epochs: int
    :param training_epochs: maximal number of iterations to run the optimizer

    :type dataset: string
    :param dataset: path of pickled dataset
    """

    datasets = load_data(dataset)

    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size

    # numpy random generator
    numpy_rng = np.random.RandomState(89677)
    print("... building the model")

    # construct the stacked denoising autoencoder class
    sda = StackedDenoisingAutoEncoder(numpy_rng=numpy_rng,
                                      n_ins=28*28,
                                      hidden_layers_sizes=[1000, 1000, 1000],
                                      n_outs=10)

    # pretraining the model
    print("... getting the pretraining functions") 
    pretraining_fns = sda.pretraining_functions(train_x=train_x,
                                                batch_size=batch_size)

    print("... pre-training the model")
    start_time = timeit.default_timer()
    corruption_levels = [.1, .2, .3]
    for i in range(sda.n_layers):
        for epoch in range(pretraining_epochs):
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            corruption=corruption_levels[i],
                                            lr=pretrain_lr))
            print("Pre-training layer %i, epoch %d, cost %f" % 
                  (i, epoch, np.mean(c, dtype='float64')))

    end_time = timeit.default_timer()

    print(("The pretrainin code for file " +
           os.path.split(__file__)[1] +
           " ran for %.2fm" % ((end_time - start_time) / 60.)), file=sys.stderr)

    # finetuning the model
    print("... getting the finetuning functions")
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets = datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print("... finetunning the model")
    # early-stopping parameters
    # look as this many examples regardless
    patience = 10 * n_train_batches
    # wait this much longer when a new best is found
    patience_increase = 2.
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience//2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                      (epoch,
                       minibatch_index + 1,
                       n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvementis good enough
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = np.mean(test_losses, dtype='float64')
                    print('epoch %i, minibatch %i/%i, test error of best model %f %%' %
                          (epoch,
                           minibatch_index + 1,
                           n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            "Optimization complete with best validation score of %f %%, "
            "on iteration %i, "
            "with test performance %f %%"
        )
        % (best_validation_loss * 100.,
           best_iter + 1,
           test_score * 100.)
    )
    print(("The training code for file " +
           os.path.split(__file__)[1] +
           " ran for %.2fm" % ((end_time - start_time) / 60.)), file=sys.stderr)
