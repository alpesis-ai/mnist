from __future__ import division, print_function, absolute_import

import tflearn
import tflearn.datasets.mnist as mnist

from models import mlp


def train(net, trainX, trainY, testX, testY):

    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, n_epoch=20,
                              validation_set=(testX, testY),
                              show_metric=True,
                              run_id="dense_model")


if __name__ == '__main__':

    trainX, trainY, testX, testY = mnist.load_data(one_hot=True)

    net = mlp()
    train(net, trainX, trainY, testX, testY)  
    
