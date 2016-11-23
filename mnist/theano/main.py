from apps.sgd_optimization import sgd_optimization
from apps.sgd_optimization import predict

from apps.mlp import classifier_mlp
from apps.lenet import classifier_lenet5
from apps.denoising_autoencoder import classifier_denoising_autoencoder
from apps.stacked_denoising_autoencoder import classifier_sda


def logistic_classifier():
    sgd_optimization()
    predict()


def mlp():
    classifier_mlp() 
 

if __name__ == '__main__':

    # logistic_classifier()
    # mlp()
    # classifier_lenet5()

    # classifier_denoising_autoencoder()
    classifier_sda()
