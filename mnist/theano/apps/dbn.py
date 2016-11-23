def classifier_dbn(finetune_lr=0.1,
                   pretraining_epochs=100,
                   pretrain_lr=0.01,
                   k=1,
                   training_epochs=1000,
                   dataset='../../data/mnist.pkl.gz',
                   batch_size=10):
    """
    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type k: int
    :param k: number of Gibbs steps in CD/PCD

    :type training_epochs: int
    :param training_epochs: maximal number of iterationsto run the optimizer

    :type dataset: string
    :param dataset: path of the pickled dataset
    
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """

    datasets = load_data(dataset)
    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size

    # numpy random generator
    numpy_rng = np.random.RandomState(123)
    print('... building the model')
    # construct the deep belief network
    dbn = DBN(numpy_rng=numpy_rng,
              n_ins=28*28,
              hidden_layers_sizes=[1000, 1000, 1000],
              n_outs=10)

    # pretraining the model
    print("... getting the pretraining functions")
    pretraining_fns = dbn.pretraining_functions(train_x=train_x,
                                                batch_size=batch_size,
                                                k=k)

    print("... pre-training the model")
    start_time = timeit.default_timer()
    # pre-train layer-wise
    for i in range(dbn.n_layers):
        for epoch in range(pretraining_epochs):
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                                            lr=pretrain_lr))

            print("Pre-training layer %i, epoch %d, cost " % (i, epoch), end=' ')
            print(np.mean(c, dtype='float64'))

    end_time = timeit.default_timer()
    print("The pretraing code for file " +
          os.path.split(__file__)[1] +
          " ran for %.2fm" % ((end_time - start_time) / 60.), file=sys.stderr)

    # finetuning the model
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model - dbn.build_finetune_functions(
        datasets=datasets,
        batch_sizes=batch_size,
        learning_rate=finetune_lr
    )

    print("... finetuning the model")
    patience = 4 * n_train_batches
    # wait this much longer when a new best is found
    patience_increase = 2.
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):
            train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
   
            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses, dtype='float64')
                print("epoch %i, minibatch %i/%i, validation error %f %%" % (
                    epoch,
                    minibatch_index + 1,
                    n_train_batches,
                    this_validation_loss * 100.
                ))
    
            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                # improve patience if loss improvement is good enough
                if (this_validation_loss < best_validation_loss * improvement_threshold):
                    patience = max(patience, iter * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = iter

                # test it on the best set
                test_losses = test_model()
                test_score = np.mean(test_losses, dtype='float64')
                print("epoch %i, minibatch %i/%i, test error of best model %f %%"
                      % (epoch,
                         minibatch_index + 1,
                         n_train_batches,
                         test_score * 100.))

        if patience <= iter:
            done_looping = True
            break

    end_time = timeit.default_timer()
    print(('Optimization complete with best validation score of %f %%, '
           'obtained at iteration %i, '
           'with test performance %f %%'
           ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The fine tuning code for file ' + os.path.split(__file__)[1] +
          ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
