import numpy as np

import theano
import theano.tensor as T

from datasets.data import load_data


def classifier_rbm(learning_rate=0.1,
                   training_epochs=15,
                   dataset='../../data/mnist.pkl.gz',
                   batch_size=20,
                   n_chains=20,
                   n_samples=10,
                   output_folder='../outputs/rbm_plots',
                   n_hidden=500):
    """ 

    :param learning_rate: learning rate used for training the RBM
    :param training_epochs: number of epochs used for training
    :param dataset: path of the pickled dataset
    :param batch_size: size of a batch used to train the RBM
    :param n_chains: number of parallel Gibbs chains to be used for sampling
    :param n_samples: number of samples to plot for each chain
    """

    datasets = load_data(dataset)
    train_x, train_y = datasets[0]
    test_x, test_y = datasets[2]

    # compute number of minibatches for training, validation, and testing
    n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()
    x = T.matrix('x')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2**30))

    # initialize storage for the persistent chain
    # state = hidden layer of chain
    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden), dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = RBM(input=x,
              n_visible=28*28,
              n_hidden=n_hidden,
              numpy_rng=rng,
              theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain,
                                         k=15)

    # training the mode
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)

    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function([index],
                                cost,
                                updates=updates,
                                givens=givens,
                                name='train_rbm')

    plotting_time = 0.
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print("Training epoch %d, cost is " % epoch, np.mean(mean_cost))

        # plot filters after each training epoch
        plotting_start = timeit.default_timer()
        # construct image from the weight matrix
        image = Image.fromarray(tile_raster_images(X=rbm.W.get_value(borrow=True).T,
                                                   img_shape=(28, 28),
                                                   tile_shape(10, 10),
                                                   tile_spacing=(1, 1)))

        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()
    pretraining_time = (end_time - start_time) - plotting_time
    print("Training took %f minutes" % (pretrainingtime / 60.))

    # sampling from RBM
    # find out the number of test samples
    number_of_test_samples = test_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(test_x.get_value(borrow=True)[test_idx : test_idx + n_chains],
        dtype=theano.config.floatX)
    )

    plot_every = 1000
    # define one step of Gibbs sampling (mf = mean-field) define a function that
    # does `plot_every` steps before returning the sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(rbm.gibbs_vhv,
                    outputs_info=[None, None, None, None, None, persistent_vis_chain],
                    n_steps=plot_every,
                    name='gibbs_vhv')

    # add to updates the shared variable that takes care of our persistent chain:
    updates.update({persistent_vis_chain: vis_samples[-1]})

    # construct the function that implements our persistent chain
    # we generate the "mean field" activations for plotting and the actual
    # samples for reinitializing the state of our persistent chain
    sample_fn = theano.function([],
                                [
                                    vis_mfs[-1],
                                    vis_samples[-1]
                                ],
                                updates=updates,
                                name='sample_fn')

    # create a space to store the image for plotting (we need to leave room
    # for the tile_spacing as well)
    image_data = np.zeros((29 * n_samples + 1, 29 * n_chains -1), dtype='unit8')
    for idx in range(n_samples):
        # generate `plot_every` intermediate samples that we discard
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print("... plotting sample %d" % idx)
        image_data[29*idx : 29*idx+28, :] = tile_raster_images(X=vis_mf,
                                                               img_shape=(28, 28),
                                                               tile_shape=(1, n_chains),
                                                               tile_spacing=(1, 1))

    # construct the image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    os.chdir('../')
