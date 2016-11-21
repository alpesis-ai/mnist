import os.path


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
