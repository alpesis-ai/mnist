from apps.sgd_optimization import sgd_optimization
from apps.sgd_optimization import predict

from apps.mlp import classifier_mlp


def logistic_classifier():
    sgd_optimization()
    predict()


def mlp():
    classifier_mlp() 
 

if __name__ == '__main__':

    # logistic_classifier()
    mlp()
